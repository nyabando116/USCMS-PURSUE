import time

import coffea.processor as processor
from coffea.analysis_tools import PackedSelection, Weights
from coffea.nanoevents import NanoAODSchema, NanoEventsFactory
from coffea.nanoevents.methods import nanoaod

NanoAODSchema.warn_missing_crossrefs = False

import pickle
import re

import awkward as ak
import numpy as np
import pandas as pd
import json
import fastjet
import dask_awkward
import hist.dask as dah
import hist

def update(events, collections):
    """Return a shallow copy of events array with some collections swapped out"""
    out = events
    for name, value in collections.items():
        out = ak.with_field(out, value, name)
    return out

# Look at ProcessorABC to see the expected methods and what they are supposed to do
class msdProcessor(processor.ProcessorABC):
    def __init__(self, isMC=False):
        ################################
        # INITIALIZE COFFEA PROCESSOR
        ################################
            
        ak.behavior.update(nanoaod.behavior)

        # Some examples of axes
        pt_axis = hist.axis.Regular(120, 0, 1200, name="pt", label=r"Jet $p_{T}$ [GeV]")
        eta_axis = hist.axis.Regular(100, -6, 6, name="eta", label=r"Jet eta")

        # Ruva can make her own axes
        ## here
        # msoftdrop_axis = hist.axis.Regular(120, 0, 1200, name="msoftdrop", label=r"Jet msoftdrop")
        # n2_axis = hist.axis.Regular(100, -6, 6, name="n2", label=r"Jet n2")
        
        
        self.make_output = lambda: { 
            # Test histogram; not needed for final analysis but useful to check things are working
            "ExampleHistogram": dah.Hist(
                pt_axis,
                eta_axis,
                # msoftdrop_axis,
                # n2_axis,
                storage=hist.storage.Weight()
            ),
        }
        
    def process(self, events):
        
        output = self.make_output()

        ##################
        # OBJECT SELECTION
        ##################

        # For soft drop studies we care about the AK8 jets
        fatjets = events.FatJet

        # Ruva can update the selection here
        jetdef = fastjet.JetDefinition(
        fastjet.cambridge_algorithm, 0.8
        )
        
        candidatejet = fatjets[(fatjets.pt > 450)
                               & (abs(fatjets.eta) < 2.5)
                               #& fatjets.isTight
                               ]

        # Let's use only one jet
        leadingjets = candidatejet[:, 0:1]

        pf = ak.flatten(leadingjets.constituents.pf, axis=1)
        cluster = fastjet.ClusterSequence(pf, jetdef)
        softdrop_zcut10_beta0 = cluster.exclusive_jets_softdrop_grooming()

        # Ruva can calculate the variables we care about here
        
        softdrop_zcut10_beta0_cluster = fastjet.ClusterSequence(softdrop_zcut10_beta0.constituents, jetdef)
        n2 = softdrop_zcut10_beta0_cluster.exclusive_jets_energy_correlator(func="nseries", npoint = 2)
        
        # Will need to use fastjet
        jetpt = leadingjets.pt
        
        jeteta = leadingjets.eta
        # jetmsoftdrop=leadingjets.msoftdrop
        # jetn2=n2
        
        # print(len(jetpt))
        # print(len(jeteta))
        # print(len(jetmsoftdrop))
        # print(len(jetn2))
        ################
        # EVENT WEIGHTS
        ################

        # Ruva can ignore this section -- it is related to how we produce MC simulation
        
        # create a processor Weights object, with the same length as the number of events in the chunk
        # weights = Weights(dask_awkward.num(events, axis=0).compute())
        weights = Weights(size=None, storeIndividual=True)
        output = self.make_output()
        output['sumw'] = ak.sum(events.genWeight)
        weights.add('genweight', events.genWeight)

        ###################
        # FILL HISTOGRAMS
        ###################
        

        output['ExampleHistogram'].fill(pt=jetpt,
                                        eta=jeteta,
                                        # msoftdrop=jetmsoftdrop,
                                        # n2=jetn2,
                                        weight=weights.weight()
                                        )
    
    
        return output

    def postprocess(self, accumulator):
        return accumulator
