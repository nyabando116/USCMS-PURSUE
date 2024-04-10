"""
Collection of utilities for corrections and systematics in processors.

Most corrections retrieved from the cms-nanoAOD repo:
See https://cms-nanoaod-integration.web.cern.ch/commonJSONSFs/
"""

import numpy as np
import awkward as ak
import gzip
import pickle
import cloudpickle
import warnings
import importlib.resources
import correctionlib
from coffea.lookup_tools.lookup_base import lookup_base
from coffea import lookup_tools
from coffea import util
import dask
import scipy.interpolate
import weakref
import dask_awkward as dak

# Important Run3 start of Run
FirstRun_2022C = 355794
FirstRun_2022D = 357487
LastRun_2022D = 359021
FirstRun_2022E = 359022
LastRun_2022F = 362180

"""
CorrectionLib files are available from: /cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration - synced daily
"""
pog_correction_path = "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/"
pog_jsons = {
    "muon": ["MUO", "muon_Z.json.gz"],
    "electron": ["EGM", "electron.json.gz"],
    "pileup": ["LUM", "puWeights.json.gz"],
    "fatjet_jec": ["JME", "fatJet_jerc.json.gz"],
    "jet_jec": ["JME", "jet_jerc.json.gz"],
    "jetveto": ["JME", "jetvetomaps.json.gz"],
    "btagging": ["BTV", "btagging.json.gz"],
}

years = {
    "2016": "2016postVFP_UL",
    "2016APV": "2016preVFP_UL",  
    "2017": "2017_UL",  
    "2018": "2018_UL",  
    "2022": "2022_Summer22",
    "2022EE": "2022_Summer22EE",
    "2023": "2023_Summer23",
    "2023BPix": "2023_Summer23BPix",
}

def get_pog_json(obj: str, year: str) -> str:
    try:
        pog_json = pog_jsons[obj]
    except:
        print(f"No json for {obj}")

    year = years[year]

    return f"{pog_correction_path}/POG/{pog_json[0]}/{year}/{pog_json[1]}"

# Experimental corrections

class JECs:
    def __init__(self, year):
        if year in ["2022", "2022EE", "2023", "2023BPix"]:
            jec_compiled = package_path + "/corrections/jec_compiled.pkl.gz"
        elif year in ["2016", "2016APV", "2017", "2018"]:
            jec_compiled = package_path + "/corrections/jec_compiled_run2.pkl.gz"
        else:
            jec_compiled = None

        self.jet_factory = {}
        self.met_factory = None

        if jec_compiled is not None:
            with gzip.open(jec_compiled, "rb") as filehandler:
                jmestuff = pickle.load(filehandler)

            self.jet_factory["ak4"] = jmestuff["jet_factory"]
            self.jet_factory["ak8"] = jmestuff["fatjet_factory"]
            self.met_factory = jmestuff["met_factory"]

    def _add_jec_variables(self, jets: JetArray, event_rho: ak.Array, isData: bool) -> JetArray:
        """add variables needed for JECs"""
        jets["pt_raw"] = (1 - jets.rawFactor) * jets.pt
        jets["mass_raw"] = (1 - jets.rawFactor) * jets.mass
        jets["event_rho"] = ak.broadcast_arrays(event_rho, jets.pt)[0]
        if not isData:
            # gen pT needed for smearing
            jets["pt_gen"] = ak.values_astype(ak.fill_none(jets.matched_gen.pt, 0), np.float32)
        return jets

    def get_jec_jets(
        self,
        events: NanoEventsArray,
        jets: FatJetArray,
        year: str,
        isData: bool = False,
        jecs: dict[str, str] | None = None,
        fatjets: bool = True,
        applyData: bool = False,
        dataset: str | None = None,
        nano_version: str = "v12",
    ) -> FatJetArray:
        """
        If ``jecs`` is not None, returns the shifted values of variables are affected by JECs.
        """

        rho = (
            events.Rho.fixedGridRhoFastjetAll
            if "Rho" in events.fields
            else events.fixedGridRhoFastjetAll
        )
        jets = self._add_jec_variables(jets, rho, isData)

        apply_jecs = ak.any(jets.pt) if (applyData or not isData) else False
        if "v12" not in nano_version:
            apply_jecs = False
        if not apply_jecs:
            return jets, None

        jec_vars = ["pt"]  # variables we are saving that are affected by JECs
        jet_factory_str = "ak4"
        if fatjets:
            jet_factory_str = "ak8"

        if self.jet_factory[jet_factory_str] is None:
            print("No factory available")
            return jets, None

        import cachetools

        jec_cache = cachetools.Cache(np.inf)

        if isData:
            if year == "2022":
                corr_key = f"{year}_runCD"
            elif year == "2022EE" and "Run2022E" in dataset:
                corr_key = f"{year}_runE"
            elif year == "2022EE" and "Run2022F" in dataset:
                corr_key = f"{year}_runF"
            elif year == "2022EE" and "Run2022G" in dataset:
                corr_key = f"{year}_runG"
            elif year == "2023":
                corr_key = "2023_runCv4" if "Run2023Cv4" in dataset else "2023_runCv123"
            elif year == "2023BPix":
                corr_key = "2023BPix_runD"
            else:
                print(dataset, year)
                print("warning, no valid dataset, JECs won't be applied to data")
                applyData = False
        else:
            corr_key = f"{year}mcnoJER" if "2023" in year else f"{year}mc"

        apply_jecs = ak.any(jets.pt) if (applyData or not isData) else False

        # fatjet_factory.build gives an error if there are no jets in event
        if apply_jecs:
            jets = self.jet_factory[jet_factory_str][corr_key].build(jets, jec_cache)

        # return only jets if no variations are given
        if jecs is None or isData:
            return jets, None

        jec_shifted_vars = {}

        for jec_var in jec_vars:
            tdict = {"": jets[jec_var]}
            if apply_jecs:
                for key, shift in jecs.items():
                    for var in ["up", "down"]:
                        tdict[f"{key}_{var}"] = jets[shift][var][jec_var]

            jec_shifted_vars[jec_var] = tdict

        return jets, jec_shifted_vars

# Jet Veto Maps
# the JERC group recommends ALL analyses use these maps, as the JECs are derived excluding these zones.
# apply to both Data and MC
# https://cms-talk.web.cern.ch/t/jet-veto-maps-for-run3-data/18444?u=anmalara
# https://cms-talk.web.cern.ch/t/jes-for-2022-re-reco-cde-and-prompt-fg/32873
def get_jetveto_event(jets: JetArray, year: str):
    """
    Get event selection that rejects events with jets in the veto map
    """

    # correction: Non-zero value for (eta, phi) indicates that the region is vetoed
    cset = correctionlib.CorrectionSet.from_file(get_pog_json("jetveto", year))
    j, nj = ak.flatten(jets), ak.num(jets)

    def get_veto(j, nj, csetstr):
        j_phi = np.clip(np.array(j.phi), -3.1415, 3.1415)
        j_eta = np.clip(np.array(j.eta), -4.7, 4.7)
        veto = cset[csetstr].evaluate("jetvetomap", j_eta, j_phi)
        return ak.unflatten(veto, nj)

    corr_str = {
        # What about Run 2?
        "2016APV": "Summer19UL16_V1",
        "2016APV": "Summer19UL16_V1",
        "2016APV": "Summer19UL17_V1",
        "2016APV": "Summer19UL18_V1",
        "2022": "Summer22_23Sep2023_RunCD_V1",
        "2022EE": "Summer22EE_23Sep2023_RunEFG_V1",
        "2023": "Summer23Prompt23_RunC_V1",
        "2023BPix": "Summer23BPixPrompt23_RunD_V1",
    }[year]

    jet_veto = get_veto(j, nj, corr_str) > 0

    event_sel = ~(ak.any((jets.pt > 15) & jet_veto, axis=1))
    return event_sel

def add_pileup_weight(weights: Weights, year: str, nPU: np.ndarray, dataset: str | None = None):
    # clip nPU from 0 to 100
    nPU = np.clip(nPU, 0, 99)
    # print(list(nPU))

    # https://twiki.cern.ch/twiki/bin/view/CMS/LumiRecommendationsRun3
    values = {}

    cset = correctionlib.CorrectionSet.from_file(get_pog_json("pileup", year))
    corr = {
        "2016APV": "Collisions16_UltraLegacy_goldenJSON",
        "2016": "Collisions16_UltraLegacy_goldenJSON",
        "2017": "Collisions17_UltraLegacy_goldenJSON",
        "2018": "Collisions18_UltraLegacy_goldenJSON",
        "2022": "Collisions2022_355100_357900_eraBCD_GoldenJson",
        "2022EE": "Collisions2022_359022_362760_eraEFG_GoldenJson",
        "2023": "Collisions2023_366403_369802_eraBC_GoldenJson",
        "2023BPix": "Collisions2023_369803_370790_eraD_GoldenJson",
    }[year]
    # evaluate and clip up to 10 to avoid large weights
    values["nominal"] = np.clip(cset[corr].evaluate(nPU, "nominal"), 0, 10)
    values["up"] = np.clip(cset[corr].evaluate(nPU, "up"), 0, 10)
    values["down"] = np.clip(cset[corr].evaluate(nPU, "down"), 0, 10)

    weights.add("pileup", values["nominal"], values["up"], values["down"])

# Run 3 only
def add_trig_weights(weights: Weights, fatjets: FatJetArray, year: str, num_jets: int = 2):
    """
    Add the trigger scale factor weights and uncertainties

    Give number of jets in pre-selection to obtain event weight
    """
    if year == "2018":
        with Path(f"{package_path}/corrections/data/fatjet_triggereff_{year}_combined.pkl").open(
            "rb"
        ) as filehandler:
            combined = pickle.load(filehandler)

        # sum over TH4q bins
        effs_txbb = combined["num"][:, sum, :, :] / combined["den"][:, sum, :, :]

        ak8TrigEffsLookup = dense_lookup(
            np.nan_to_num(effs_txbb.view(flow=False), 0), np.squeeze(effs_txbb.axes.edges)
        )

        fj_trigeffs = ak8TrigEffsLookup(
            pad_val(fatjets.Txbb, num_jets, axis=1),
            pad_val(fatjets.pt, num_jets, axis=1),
            pad_val(fatjets.msoftdrop, num_jets, axis=1),
        )

        combined_trigEffs = 1 - np.prod(1 - fj_trigeffs, axis=1)

        weights.add("trig_effs", combined_trigEffs)
        return

    # TODO: replace year
    year = "2022EE"
    jet_triggerSF = correctionlib.CorrectionSet.from_file(
        package_path + f"/corrections/data/fatjet_triggereff_{year}_combined_nodijet.json"
    )

    fatjets_xbb = pad_val(fatjets.Txbb, num_jets, axis=1)
    fatjets_pt = pad_val(fatjets.pt, num_jets, axis=1)

    nom_data = jet_triggerSF[f"fatjet_triggereff_{year}_data"].evaluate(
        "nominal", fatjets_pt, fatjets_xbb
    )
    nom_mc = jet_triggerSF[f"fatjet_triggereff_{year}_MC"].evaluate(
        "nominal", fatjets_pt, fatjets_xbb
    )

    nom_data_up = jet_triggerSF[f"fatjet_triggereff_{year}_data"].evaluate(
        "stat_up", fatjets_pt, fatjets_xbb
    )
    nom_mc_up = jet_triggerSF[f"fatjet_triggereff_{year}_MC"].evaluate(
        "stat_up", fatjets_pt, fatjets_xbb
    )

    nom_data_dn = jet_triggerSF[f"fatjet_triggereff_{year}_data"].evaluate(
        "stat_dn", fatjets_pt, fatjets_xbb
    )
    nom_mc_dn = jet_triggerSF[f"fatjet_triggereff_{year}_MC"].evaluate(
        "stat_dn", fatjets_pt, fatjets_xbb
    )

    # calculate trigger weight per event and take ratio from data/MC
    combined_eff_data = 1 - np.prod(1 - nom_data, axis=1)
    combined_eff_mc = 1 - np.prod(1 - nom_mc, axis=1)
    sf = combined_eff_data / combined_eff_mc

    sf_up = (1 - np.prod(1 - nom_data_up, axis=1)) / (1 - np.prod(1 - nom_mc_up, axis=1))
    sf_dn = (1 - np.prod(1 - nom_data_dn, axis=1)) / (1 - np.prod(1 - nom_mc_dn, axis=1))

    weights.add(f"trigsf_{num_jets}jet", sf, sf_up, sf_dn)

# Muon triggers are garbage, please fix
def add_mutriggerSF(weights, leadingmuon, year, selection):
    def mask(w):
        return ak.where(selection.all('onemuon'), w, 1.)
    mu_pt = ak.fill_none(leadingmuon.pt, 0.)
    mu_eta = ak.fill_none(abs(leadingmuon.eta), 0.)
    nom = mask(dask_compliance(compiled[f'{year}_mutrigweight_pt_abseta'])(mu_pt, mu_eta))
    shift = mask(dask_compliance(compiled[f'{year}_mutrigweight_pt_abseta_mutrigweightShift'])(mu_pt, mu_eta))
    weights.add('mu_trigger', nom, shift, shift=True)
#     abcd

def add_muon_weights(weights: Weights, muons: MuonArray, year: str):

    def mask(w):
        return ak.where(selection.all('onemuon'), w, 1.)

    cset = correctionlib.CorrectionSet.from_file(get_pog_json("muon", year))
    
    mu_pt = muons.pt
    mu_eta = muons.eta

    # Muon ID
    id_nom = mask(cset['NUM_MediumPromptID_DEN_TrackerMuons'].evaluate(years[year], mu_eta, mu_pt, "sf"))
    id_up = mask(cset['NUM_MediumPromptID_DEN_TrackerMuons'].evaluate(years[year],mu_eta, mu_pt, "systup"))
    id_down = mask(cset['NUM_MediumPromptID_DEN_TrackerMuons'].evaluate(years[year],mu_eta, mu_pt, "systdown"))
    weights.add('mu_idweight', id_nom, id_up, id_down)

    # Muon isolation
    iso_nom = mask(cset['NUM_LooseRelIso_DEN_MediumPromptID'].evaluate(years[year], mu_eta, mu_pt, "sf"))
    iso_up = mask(cset['NUM_LooseRelIso_DEN_MediumPromptID'].evaluate(years[year],mu_eta, mu_pt, "systup"))
    iso_down = mask(cset['NUM_LooseRelIso_DEN_MediumPromptID'].evaluate(years[year],mu_eta, mu_pt, "systdown"))
    weights.add('mu_isoweight', iso_nom, iso_up, iso_down)
    
def n2ddt_shift(fatjets, year='2017'):
    #need to setattrs _dask_future and _weakref
    compiled[f'{year}_n2ddt_rho_pt'] = dask_compliance(compiled[f'{year}_n2ddt_rho_pt'])
    return compiled[f'{year}_n2ddt_rho_pt'](fatjets.qcdrho, fatjets.pt)
       
class SoftDropWeight(lookup_base):
    def __init__(self):
        dask_future = dask.delayed(
            self, pure=True, name=f"softdropweight-{dask.base.tokenize(self)}"
        ).persist()
        super().__init__(dask_future)
        
    def _evaluate(self, pt, eta, **kwargs):
        gpar = ak.Array([1.00626, -1.06161, 0.0799900, 1.20454])
        cpar = ak.Array([1.09302, -0.000150068, 3.44866e-07, -2.68100e-10, 8.67440e-14, -1.00114e-17])
        fpar = ak.Array([1.27212, -0.000571640, 8.37289e-07, -5.20433e-10, 1.45375e-13, -1.50389e-17])
        genw = gpar[0] + gpar[1]*np.power(pt*gpar[2], -gpar[3])
        cenweight = np.polyval(cpar[::-1], pt)
        forweight = np.polyval(fpar[::-1], pt)
        weight = np.where(np.abs(eta) < 1.3, cenweight, forweight)
        return genw*weight

def corrected_msoftdrop(fatjets):
    _softdrop_weight = SoftDropWeight()
    sf = _softdrop_weight(fatjets.pt, fatjets.eta)
    sf = np.maximum(1e-5, sf)
    dazsle_msd = (fatjets.subjets * (1 - fatjets.subjets.rawFactor)).sum()
    return dazsle_msd.mass * sf

def build_lumimask(filename):
    from coffea.lumi_tools import LumiMask
    with importlib.resources.path("boostedhiggs.data", filename) as path:
        return LumiMask(path)

lumiMasks = {
    '2016': build_lumimask('Cert_271036-284044_13TeV_23Sep2016ReReco_Collisions16_JSON.txt'),
    '2017': build_lumimask('Cert_294927-306462_13TeV_EOY2017ReReco_Collisions17_JSON_v1.txt'),
    '2018': build_lumimask('Cert_314472-325175_13TeV_17SeptEarlyReReco2018ABC_PromptEraD_Collisions18_JSON.txt'),
}

# Theory corrections
def add_scalevar(weights, var_weights):

    docstring = var_weights.__doc__

    nweights = len(weights.weight())
    nom = np.ones(nweights)

    for i in range(0,9):
        weights.add('scalevar_'+str(i), nom, var_weights[:,i])

def add_pdf_weight(weights, pdf_weights):

    docstring = pdf_weights.__doc__

    nweights = len(weights.weight())
    nom = np.ones(nweights)

    for i in range(0,103):
        weights.add('PDF_weight_'+str(i), nom, pdf_weights[:,i])
        
def add_ps_weight(weights, ps_weights):
    nom = ak.ones_like(weights.weight())
    up_isr = ak.ones_like(weights.weight())
    down_isr = ak.ones_like(weights.weight())
    up_fsr = ak.ones_like(weights.weight())
    down_fsr = ak.ones_like(weights.weight())

    if ps_weights is not None:
        #if len(ps_weights[0]) == 4:
        if ps_weights.__doc__ == 'PS weights (w_var / w_nominal);   [0] is ISR=2 FSR=1; [1] is ISR=1 FSR=2[2] is ISR=0.5 FSR=1; [3] is ISR=1 FSR=0.5;':
            up_isr = ps_weights[:, 0]
            down_isr = ps_weights[:, 2]
            up_fsr = ps_weights[:, 1]
            down_fsr = ps_weights[:, 3]
    #        up = np.maximum.reduce([up_isr, up_fsr, down_isr, down_fsr])
    #        down = np.minimum.reduce([up_isr, up_fsr, down_isr, down_fsr])
        else:
            warnings.warn(f"PS weight vector has length {len(ps_weights[0])}")
            

    weights.add('UEPS_ISR', nom, up_isr, down_isr)
    weights.add('UEPS_FSR', nom, up_fsr, down_fsr)

def get_vpt(genpart, check_offshell=False):
    """Only the leptonic samples have no resonance in the decay tree, and only
    when M is beyond the configured Breit-Wigner cutoff (usually 15*width)
    """
    boson = ak.firsts(
        genpart[
            ((genpart.pdgId == 23) | (abs(genpart.pdgId) == 24))
            & genpart.hasFlags(["fromHardProcess", "isLastCopy"])
        ]
    )
    if check_offshell:
        offshell = genpart[
            genpart.hasFlags(["fromHardProcess", "isLastCopy"])
            & ak.is_none(boson)
            & (abs(genpart.pdgId) >= 11)
            & (abs(genpart.pdgId) <= 16)
        ].sum()
        return ak.where(ak.is_none(boson.pt), offshell.pt, boson.pt)
    return np.array(ak.fill_none(boson.pt, 0.0))

def add_VJets_kFactors(weights, genpart, dataset):
    """Revised version of add_VJets_NLOkFactor, for both NLO EW and ~NNLO QCD"""

    common_systs = [
        "d1K_NLO",
        "d2K_NLO",
        "d3K_NLO",
        "d1kappa_EW",
    ]
    zsysts = common_systs + [
        "Z_d2kappa_EW",
        "Z_d3kappa_EW",
    ]
    wsysts = common_systs + [
        "W_d2kappa_EW",
        "W_d3kappa_EW",
    ]

    def add_systs(systlist, qcdcorr, ewkcorr, vpt):
        ewknom = ewkcorr.evaluate("nominal", vpt)
        weights.add("vjets_nominal", qcdcorr * ewknom if qcdcorr is not None else ewknom)
        ones = np.ones_like(vpt)
        for syst in systlist:
            weights.add(syst, ones, ewkcorr.evaluate(syst + "_up", vpt) / ewknom, ewkcorr.evaluate(syst + "_down", vpt) / ewknom)

    if "ZJetsToQQ_HT" in dataset and "TuneCUETP8M1" in dataset:
        vpt = get_vpt()
        qcdcorr = vjets_kfactors["Z_MLM2016toFXFX"].evaluate(vpt)
        ewkcorr = vjets_kfactors["Z_FixedOrderComponent"]
        add_systs(zsysts, qcdcorr, ewkcorr, vpt)
    elif "WJetsToQQ_HT" in dataset and "TuneCUETP8M1" in dataset:
        vpt = get_vpt()
        qcdcorr = vjets_kfactors["W_MLM2016toFXFX"].evaluate(vpt)
        ewkcorr = vjets_kfactors["W_FixedOrderComponent"]
        add_systs(wsysts, qcdcorr, ewkcorr, vpt)
    elif "ZJetsToQQ_HT" in dataset and "TuneCP5" in dataset:
        vpt = get_vpt()
        qcdcorr = vjets_kfactors["Z_MLM2017toFXFX"].evaluate(vpt)
        ewkcorr = vjets_kfactors["Z_FixedOrderComponent"]
        add_systs(zsysts, qcdcorr, ewkcorr, vpt)
    elif "WJetsToQQ_HT" in dataset and "TuneCP5" in dataset:
        vpt = get_vpt()
        qcdcorr = vjets_kfactors["W_MLM2017toFXFX"].evaluate(vpt)
        ewkcorr = vjets_kfactors["W_FixedOrderComponent"]
        add_systs(wsysts, qcdcorr, ewkcorr, vpt)
    elif ("DY1JetsToLL_M-50" in dataset or "DY2JetsToLL_M-50" in dataset) and "amcnloFXFX" in dataset:
        vpt = get_vpt(check_offshell=True)
        ewkcorr = vjets_kfactors["Z_FixedOrderComponent"]
        add_systs(zsysts, None, ewkcorr, vpt)
    elif ("W1JetsToLNu" in dataset or "W2JetsToLNu" in dataset) and "amcnloFXFX" in dataset:
        vpt = get_vpt(check_offshell=True)
        ewkcorr = vjets_kfactors["W_FixedOrderComponent"]
        add_systs(wsysts, None, ewkcorr, vpt)

def add_HiggsEW_kFactors(weights, genpart, dataset):
    """EW Higgs corrections"""
    def get_hpt():
        boson = ak.firsts(genpart[
            (genpart.pdgId == 25)
            & genpart.hasFlags(["fromHardProcess", "isLastCopy"])
        ])
        return np.array(ak.fill_none(boson.pt, 0.))

    if "VBF" in dataset:
        hpt = get_hpt()
        ewkcorr = hew_kfactors["VBF_EW"]
        ewknom = ewkcorr.evaluate(hpt)
        weights.add("VBF_EW", ewknom)

    if "WplusH" in dataset or "WminusH" in dataset or "ZH" in dataset:
        hpt = get_hpt()
        ewkcorr = hew_kfactors["VH_EW"]
        ewknom = ewkcorr.evaluate(hpt)
        weights.add("VH_EW", ewknom)

    if "ttH" in dataset:
        hpt = get_hpt()
        ewkcorr = hew_kfactors["ttH_EW"]
        ewknom = ewkcorr.evaluate(hpt)
        weights.add("ttH_EW", ewknom)
