import time

import coffea.processor as processor
import hist
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
        
        self.make_output = lambda: { 
            # Test histogram; not needed for final analysis but useful to check things are working
            "ExampleHistogram": hist.Hist(
                pt_axis,
                eta_axis,
                storage=hist.storage.Weight()
            ),
            "EventCount": processor.value_accumulator(int),
        }
        
    def process(self, events):
        
        output = self.make_output()

        ##################
        # OBJECT SELECTION
        ##################

        # For soft drop studies we care about the AK8 jets
        fatjets = events.FatJet

        # Ruva can update the selection here
        candidatejet = fatjets[(fatjets.pt > 450)
                               & (abs(fatjets.eta) < 2.5)
                               #& fatjets.isTight
                               ]

        # Let's use only one jet
        leadingjets = candidatejet[:, 0:1]

        # Ruva can calculate the variables we care about here
        # Will need to use fastjet
        jetpt = leadingjets.pt
        jeteta = leadingjets.eta
        
        ################
        # EVENT WEIGHTS
        ################

        # Ruva can ignore this section -- it is related to how we produce MC simulation
        
        # create a processor Weights object, with the same length as the number of events in the chunk
        weights = Weights(len(events))
        weights.add('genweight', events.genWeight)

        ###################
        # FILL HISTOGRAMS
        ###################
        

        output['ExampleHistogram'].fill(pt=jetpt,
                                        eta=jeteta,
                                        weight=weights.weight()
                                        )
    
        output["EventCount"] = len(events)
    
        return output

    def postprocess(self, accumulator):
        return accumulator
