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

        #n2 axes beta
        n2b_axis = hist.axis.Regular(10, 0, 1, name="n2b", label=r"Jet n2b")
        n2b2_axis = hist.axis.Regular(10, 0, 1, name="n2b2", label=r"Jet n2b2")
        n2b3_axis = hist.axis.Regular(10, 0, 1, name="n2b3", label=r"Jet n2b3")

        #mass axes beta
        msoftdrop1_axis = hist.axis.Regular(36, 0, 252, name="msoftdrop1", label=r"Jet msoftdrop")
        msoftdrop2_axis = hist.axis.Regular(36, 0, 252, name="msoftdrop2", label=r"Jet msoftdrop")
        msoftdrop3_axis = hist.axis.Regular(36, 0, 252, name="msoftdrop3", label=r"Jet msoftdrop")

        #n2 axes zcut
        n2z1_axis = hist.axis.Regular(10, 0, 1, name="n2z1", label=r"Jet n2b")
        n2z2_axis = hist.axis.Regular(10, 0, 1, name="n2z2", label=r"Jet n2b2")

        #mass axes zcut
        msoftdropz1_axis = hist.axis.Regular(36, 0, 252, name="msoftdropz1", label=r"Jet msoftdrop z1")
        msoftdropz2_axis = hist.axis.Regular(36, 0, 252, name="msoftdropz2", label=r"Jet msoftdrop z2")
        
        
        self.make_output = lambda: { 
            # Test histogram; not needed for final analysis but useful to check things are working
            "ExampleHistogram": dah.Hist(
                pt_axis,
                eta_axis,
                msoftdrop_axis,
                n2_axis,
                storage=hist.storage.Weight()
            ),
            "ExampleHistogram1": dah.Hist(
                n2b_axis,
                msoftdrop1_axis,
                storage=hist.storage.Weight()
            ),  
            "ExampleHistogram2": dah.Hist(
                n2b2_axis,
                msoftdrop2_axis,
                storage=hist.storage.Weight()
            ), 
            "ExampleHistogram3": dah.Hist(
                n2b3_axis,
                msoftdrop3_axis,
                storage=hist.storage.Weight()
            ),
            "ExampleHistogram4": dah.Hist(
                n2z1_axis,
                msoftdropz1_axis,
                storage=hist.storage.Weight()
            ),  
            "ExampleHistogram5": dah.Hist(
                n2z2_axis,
                msoftdropz2_axis,
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
        jeteta = ak.firsts(leadingjets.eta)
        jetmsoftdrop= ak.firsts(leadingjets.msoftdrop)

        jetdef = fastjet.JetDefinition(
        fastjet.cambridge_algorithm, 0.8
        )

        pf = ak.flatten(leadingjets.constituents.pf, axis=1)
        
        # cluster = fastjet.ClusterSequence(pf, jetdef)
        softdrop_zcut10_beta0 = fastjet.ClusterSequence(pf, jetdef).exclusive_jets_softdrop_grooming(beta=0)

        # # Ruva can calculate the variables we care about here
        softdrop_zcut10_beta0_cluster = fastjet.ClusterSequence(softdrop_zcut10_beta0.constituents, jetdef)
        n2 = softdrop_zcut10_beta0_cluster.exclusive_jets_energy_correlator(func="nseries", npoint = 2)
        jetn2=n2

        #define function of n2 here
        def n2(beta, zcut):
            softdrop = fastjet.ClusterSequence(pf, jetdef).exclusive_jets_softdrop_grooming(beta=beta, symmetry_cut=zcut)
            softdrop_cluster = fastjet.ClusterSequence(softdrop.constituents, jetdef)
            n2=softdrop_cluster.exclusive_jets_energy_correlator(func="nseries", npoint=2)
            return n2

        jetn2b = n2(beta=1, zcut=0.1)
        jetn2b2 = n2(beta=2, zcut=0.1)
        jetn2b3 = n2(beta=-0.5, zcut=0.1)

        jetn2z1 = n2(beta=0, zcut=0.05)
        jetn2z2 = n2(beta=0, zcut=0.20)

       #define function for msoftdrop
        def msoftdrop(beta, zcut):
            softdrop = fastjet.ClusterSequence(pf, jetdef).exclusive_jets_softdrop_grooming(beta=beta, symmetry_cut=zcut)
            jetmsoftdrop = softdrop.msoftdrop.compute() #delayed?
            return jetmsoftdrop
       
        jetmsoftdropz1 = msoftdrop(beta=0, zcut=0.05)
        jetmsoftdropz2 = msoftdrop(beta=0, zcut=0.20)
        jetmsoftdrop1=msoftdrop(beta=1, zcut = 0.10)
        jetmsoftdrop2=msoftdrop(beta= 2, zcut = 0.10)
        jetmsoftdrop3=msoftdrop(beta=-0.5, zcut=0.10)

 
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
     

        output['ExampleHistogram'].fill(pt=normalize(jetpt),
                                        eta=normalize(jeteta),
                                        msoftdrop=normalize(jetmsoftdrop),
                                        n2=normalize(jetn2),
                                        # weight=weights
                                        weight=weights.weight()[jetpt is not None]
                                        ),
        
        output['ExampleHistogram1'].fill(n2b=normalize(jetn2b),
                                         msoftdrop1=normalize(jetmsoftdrop1),
                                         # weight=weights
                                         weight=weights.weight()[jetpt is not None]
                                  ),
        output['ExampleHistogram2'].fill(n2b2=normalize(jetn2b2),
                                         msoftdrop2=normalize(jetmsoftdrop2),
                                         weight=weights.weight()[jetpt is not None]
                                        ),
        output['ExampleHistogram3'].fill(n2b3=normalize(jetn2b3),
                                         msoftdrop3=normalize(jetmsoftdrop3),
                                         weight=weights.weight()[jetpt is not None]
                                        ),
        output['ExampleHistogram4'].fill(n2z1=normalize(jetn2z1),
                                         msoftdropz1=normalize(jetmsoftdropz1),
                                         weight=weights.weight()[jetpt is not None]
                                        ),
        output['ExampleHistogram5'].fill(n2z2=normalize(jetn2z2),
                                         msoftdropz2=normalize(jetmsoftdropz2),
                                         weight=weights.weight()[jetpt is not None]
                                        )
    
        return output

    def postprocess(self, accumulator):
        return accumulator
