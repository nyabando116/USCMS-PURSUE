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

# Look at ProcessorABC to see the expected methods and what they are supposed to do
class msdProcessor(processor.ProcessorABC):
    def __init__(self, isMC=False):
        ################################
        # INITIALIZE COFFEA PROCESSOR
        ################################
            
        # Some examples of axes
        pt_axis = hist.axis.Regular(15, 450, 1200, name="pt", label=r"Jet $p_{T}$ [GeV]")
        eta_axis = hist.axis.Regular(12, -6, 6, name="eta", label=r"Jet eta")

        # Ruva can make her own axes
        ## here
        msoftdrop_axis = hist.axis.Regular(36, 0, 252, name="msoftdrop", label=r"Jet msoftdrop")
        n2_axis = hist.axis.Regular(10, 0, 1, name="n2", label=r"Jet n2")
        
        self.make_output = lambda: { 
            # Test histogram; not needed for final analysis but useful to check things are working
            "ExampleHistogram": dah.Hist(
                pt_axis,
                eta_axis,
                msoftdrop_axis,
                n2_axis,
                storage=hist.storage.Weight()
            ),
        }#fill with beta 0, make a second histogram b=1,2,-.5...remove eta if there are memory issues 
        
    def process(self, events):
        
        output = self.make_output()

        ##################
        # OBJECT SELECTION
        ##################

        # For soft drop studies we care about the AK8 jets
        fatjets = events.FatJet
        
        candidatejet = fatjets[(fatjets.pt > 450)
                               & (abs(fatjets.eta) < 2.5)
                               #& fatjets.isTight
                               ]

        # Let's use only one jet
        leadingjets = candidatejet[:, 0:1]

        jetpt = ak.firsts(leadingjets.pt)    
        # print(jetpt.compute())
        # print(len(ak.flatten(jetpt.compute(), axis=0)))
        jeteta = ak.firsts(leadingjets.eta)
        jetmsoftdrop= ak.firsts(leadingjets.msoftdrop)

        jetdef = fastjet.JetDefinition(
        fastjet.cambridge_algorithm, 0.8
        )
        #changes here

        pf = ak.flatten(leadingjets.constituents.pf, axis=1)
        # print(len(pf.compute()))
        
        # cluster = fastjet.ClusterSequence(pf, jetdef)
        softdrop_zcut10_beta0 = fastjet.ClusterSequence(pf, jetdef).exclusive_jets_softdrop_grooming()

        # # Ruva can calculate the variables we care about here
        softdrop_zcut10_beta0_cluster = fastjet.ClusterSequence(softdrop_zcut10_beta0.constituents, jetdef)
        
        
        n2 = softdrop_zcut10_beta0_cluster.exclusive_jets_energy_correlator(func="nseries", npoint = 2)
        ##start here with n2
        jetn2=n2
        # print(jetn2.compute())

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
        def normalize(val, cut = None):
            if cut is None:
                ar = ak.flatten(val,axis=0)
                return ar
            else:
                ar = ak.flatten(val)
                return ar 
         ##use dask???     
        # mask = dask.num(jetpt) & dask.num(jeteta) & dask.num(jetmsoftdrop) & dask.num(jetn2)
        # jetpt = jetpt[mask]
        # jeteta = jeteta[mask]
        # jetmsoftdrop = jetmsoftdrop[mask]
        # jetn2 = jetn2[mask]
        # weights = weights.weight()[mask]

        output['ExampleHistogram'].fill(pt=normalize(jetpt),
                                        eta=normalize(jeteta),
                                        msoftdrop=normalize(jetmsoftdrop),
                                        n2=normalize(jetn2),
                                        # weight=weights
                                        weight=weights.weight()[jetpt is not None]
                                        )
    
    
        return output

    def postprocess(self, accumulator):
        return accumulator
