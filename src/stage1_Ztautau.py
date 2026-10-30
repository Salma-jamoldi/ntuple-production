"""Z -> tau tau events with one 1-prong and one 3-prong hemisphere (the two jets
of an exclusive ee_kt clustering), a pion sample for dE/dx studies. Run from
this directory (src/), on data and on MC:

    fccanalysis run stage1_Ztautau.py -i <data files> -o <out.root> -j 16 -- --doData
    fccanalysis run stage1_Ztautau.py -i <tau MC files> -o <out.root> -j 16

The input is an EDM4hep file of a converter version that writes ALPHA's dE/dx
collections (dEdxWiresPID, dEdxPadsPID; on MC also dEdxWiresQDEDXMPID,
dEdxWiresQDEDXMMCLink, trackMCLink and the MCParticles parents) and the LEP
energy (RunInformation). An input without them (an older conversion) stops
with an error naming them, before any event is read; a RunInformation
without the solenoid field (a converter before the field was exported) stops
the job with an error in the event loop.

Options after `--`: --doData (data: run selection, no truth branches), --year,
--noRunList and --excludeRuns RUN [RUN ...] (the run selection of stage1, from
run_list.py; data only). An unknown option is an error.

Particles, jets and tracks are stage1's: the jets are clustered from
AnalysisParticles (energy-flow types 0 to 5, without the photons and neutral
hadrons below 1 GeV); the track of a particle and its track state are read
through the particle -> track relation. Selection, in this order: exactly two
jets; one jet with 1 and one with 3 charged constituents (types 0, 1, 2);
no electron or muon in the 3-prong jet; every charged constituent with
|d0| < 2 cm and |z0| < 10 cm (track parameters at the origin) and with TPC
hits; every constituent above 1 GeV and the two jets together above 10 % of
the centre-of-mass energy, 2 x EVT_Ebeam.

Output branches. pfcand_* are per jet and per constituent (jet index first);
pfcand3_* / pfcand1_* are copies for the 3-prong / 1-prong jet only of every
pfcand_* branch except the VDET and ITC hit counts and the track chi2, ndof
and chi2/ndf; jet_* and tau_* have one entry per jet. Unit in [], value when
undefined in ().
  Event
    event_number, run_number           event and run number
    event_class                        ALEPH event classes of the event (class N = bit N-1 of ClassBitset)
    EVT_Ebeam                          LEP beam energy [GeV]
  Selection quantities
    jet_nCharged                       charged constituents (types 0, 1, 2) per jet
    jet_max_absd0, jet_max_absz0       largest |d0|, |z0| of the jet's charged constituents [cm]
    jet_min_TPChits_charged            fewest TPC hits of the jet's charged constituents (-1)
    tau_visible_mass, tau_charged_mass mass of the jet, of its charged constituents [GeV]
    tau3_visible_mass, tau3_charged_mass   the same for the 3-prong jet
  Constituents
    pfcand_isMu, _isEl, _isChargedHad, _isGamma, _isNeutralHad   1 = energy-flow type 2, 1, 0, 4, 5
    pfcand_e, _p, _px, _py, _pz [GeV], pfcand_charge [e]
    pfcand_d0, pfcand_z0               track parameters at the origin, sign of stage1's pfcand_d0 [cm] (-9)
    pfcand_nTrackHits_VDET, _ITC, _TPC hits of the constituent's track (-1)
    pfcand_trackChi2, _trackNdof, _trackChi2Norm   track fit chi2, degrees of freedom, chi2/ndf (-1)
    pfcand_trackIdx                    index of the constituent's track in Tracks (-1)
    pfcand_trkBaseline                 1 = the track passes stage1's baseline track selection, 0 = it
                                       does not (-1)
  Constituent dE/dx, normalised to about 1 for a minimum-ionising particle; -999 = not available
    pfcand_dEdx_{wires,pads}_...       TPC wires (dEdxWiresPID, ALPHA's QDEDX) / pads (dEdxPadsPID, QPADX),
                                       h = ele, mu, pi, kaon, proton:
      _value, _nSamples                measured dE/dx and its number of samples (-999)
      _expected_<h>, _sigma_<h>        ALPHA's expected dE/dx and resolution for hypothesis h,
                                       momentum error included (-999; also for a hypothesis ALPHA
                                       cannot evaluate)
      _pull_<h>                        (value - expected_<h>) / sigma_<h> (-999)
    pfcand_dEdx_wiresQDEDXM_{value,expected_<h>,sigma_<h>,pull_<h>}
                                       MC only: the same from ALPHA's parametrised simulation QDEDXM
  MC only (PDG codes: +211 = pi+, but +11 = e-)
    MC_pdg, MC_status, MC_momentum_{x,y,z}, MC_vertex_{x,y,z}   all MC particles [GeV, cm]
    pfcand_mc_pdg                      PDG code of ALPHA's best-match MC particle of the constituent's
                                       track (KBESTM); 0 = none, -999 = no track or no ALPHA MC banks
    pfcand_mc_index                    its index in MCParticles (-1 = none; -999 as pfcand_mc_pdg)
    pfcand_mc_momentum                 its momentum [GeV] (-999)
    jet3_mc_index                      pfcand_mc_index of the 3-prong jet
    jet3_same_vertex                   the best-match particles of the 3-prong jet share their production
                                       vertex (false with fewer than two of them)
    truth_tau3pi_mass                  mass of those particles and of the tau neutrino produced at their
                                       vertex [GeV] (-1)
"""
from argparse import ArgumentParser
import os 
import sys
# fccanalysis loads this file by path, so this directory's modules are made importable here
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_list

MIN_CONSTITUENT_E = 1.      # minimum energy of every constituent of both jets [GeV]
# ALPHA dE/dx hypotheses h, in the order of AlephVars in analyzer_alephvars.h (as in stage1)
DEDX_HYPOS = ("ele", "mu", "pi", "kaon", "proton")
# converter collections the dE/dx, LEP-energy and truth branches read; the second set MC only
ALEPH_COLLS = ("dEdxWiresPID", "_dEdxWiresPID_parameters", "dEdxPadsPID", "_dEdxPadsPID_parameters", "RunInformation")
ALEPH_TRUTH_COLLS = ("dEdxWiresQDEDXMPID", "_dEdxWiresQDEDXMPID_parameters", "_dEdxWiresQDEDXMMCLink_from",
                     "_dEdxWiresQDEDXMMCLink_to", "_MCParticles_parents", "trackMCLink", "_trackMCLink_from",
                     "_trackMCLink_to")


class Analysis():
    # ------------------------------------------------------------------
    # Argument handling / I-O configuration
    # ------------------------------------------------------------------
    def __init__(self, cmdline_args):
        parser = ArgumentParser(description='Z->tautau arguments (after --)')
        parser.add_argument('--doData', action='store_true',
                            help='Run on data, instead of MC (which is the default behaviour).')
        parser.add_argument('--year', default='1994',
                            help='Year of the run list (data).')
        parser.add_argument('--excludeRuns', nargs='+', action='extend', default=[], type=run_list.run_number, metavar='RUN',
                            help='Data only: drop these run numbers in addition to the run list.')
        parser.add_argument('--noRunList', action='store_true',
                            help='Data only: keep every run instead of the run list (--excludeRuns still applies).')
        self.ana_args, unknown = parser.parse_known_args(cmdline_args['remaining'])
        if unknown:
            print(f"----> ERROR: unrecognised arguments: {' '.join(unknown)}")
            sys.exit(1)
        if not self.ana_args.doData and (self.ana_args.excludeRuns or self.ana_args.noRunList):
            print("----> ERROR: --excludeRuns and --noRunList apply to data only (--doData).")
            sys.exit(1)

        """
        unlike the Z->qq template, there is no truth-flavour split here
        (Z->tautau MC is not sub-divided by quark flavour), so the
        --MCtype/--MCflavour/outnames_dict machinery from the qq analysis
        has been dropped. If you do want to label 1-prong vs 3-prong output
        separately, that is done downstream from jet_nCharged, not here.
        """

        # input and output come from the fccanalysis command line: no process list
        if not cmdline_args.get('input') and not cmdline_args.get('input_file_list'):
            sys.exit("stage1_Ztautau.py: give the input file(s) with -i and the output file with -o")
        self.n_threads = 16      # -j on the command line overrides
 
        self.include_paths = ["aleph_units.h", "analyzer.h", "analyzer_alephvars.h", "analyzer_ztautau.h"]

    #------------------------------------------------------------------
    # Event loop
    # ------------------------------------------------------------------
    def analyzers(self, df):
 
        # converter collections of the dE/dx, LEP-energy and truth branches, checked before any event is read
        have = set(str(c) for c in df.GetColumnNames())
        absent = [c for c in ALEPH_COLLS + (() if self.ana_args.doData else ALEPH_TRUTH_COLLS) if c not in have]
        if absent:
            print(f"----> ERROR: the input lacks the converter collections {', '.join(absent)}: it comes from an "
                  "older converter version; convert it again with a current one.")
            sys.exit(1)

        coll = {
            "GenParticles": "MCParticles",
            "PFParticles": "RecoParticles",
            "PFTracks": "EFlowTrack",
            "PFPhotons": "EFlowPhoton",
            "PFNeutralHadrons": "EFlowNeutralHadron",
            "TrackState": "_Tracks_trackStates",
            "TrackerHits": "TrackerHits",
            "CalorimeterHits": "CalorimeterHits",
            "PathLength": "EFlowTrack_L",
            "Bz": "magFieldBz",
        }
        AV = "FCCAnalyses::AlephVars"
 
        # run selection of stage1 (data)
        if self.ana_args.doData:
            df = run_list.filter_runs(df, self.ana_args.year, self.ana_args.excludeRuns, self.ana_args.noRunList)

        df = df.Define("event_number", "EventHeader.eventNumber")
        df = df.Define("run_number", "EventHeader.runNumber")
        df = df.Define("event_class", "AlephSelection::bitsetToIndices(ClassBitset)")
        df = df.Define("EVT_Ebeam", f"{AV}::alephRunInfo(RunInformation, true).beamEnergy")

        """
        MC truth info (kept for truth-matching / purity checks only -
        the kinematic and dE/dx variables below are all reco-level and
        identical in structure between data and MC).
        """

        #if self.ana_args.doData:
            #df = df.Filter("AlephSelection::sel_class_filter(16)(ClassBitset)   || AlephSelection::sel_class_filter(17)(ClassBitset) ")
            #df = df.Filter("AlephSelection::sel_class_filter(16)(ClassBitset) ") #basically each events are classified into distinct categories; this one says that we have hadrons in the final state so Z->qq
            #df = df.Define("jetPID", "-999")
            
        if not self.ana_args.doData:
            df = df.Define("MC_pdg", "MCParticle::get_pdg(MCParticles)")
            df = df.Define("MC_status", "MCParticle::get_genStatus(MCParticles)")
            df = df.Define("MC_momentum_x", "MCParticle::get_px(MCParticles)")
            df = df.Define("MC_momentum_y", "MCParticle::get_py(MCParticles)")
            df = df.Define("MC_momentum_z", "MCParticle::get_pz(MCParticles)")
            # ALPHA's best-match MC particle of every track (KBESTM), arrays over Tracks
            df = df.Define("trkBestMatch", f"{AV}::trackBestMatchTruth(Tracks.size(), dEdxWiresQDEDXMPID.empty(), "
                                           "_dEdxWiresQDEDXMMCLink_from, _dEdxWiresQDEDXMMCLink_to, MCParticles, "
                                           "_MCParticles_parents, _trackMCLink_from, _trackMCLink_to, "
                                           "trackMCLink.weight)")

        # Particle collections of stage1: filtered copies of RecoParticles with their ParticleID and
        # particle -> track relation; AnalysisParticles = energy-flow types 0 to 5 without the photons and
        # neutral hadrons below AlephSelection::kNeutralMinE
        df = df.Define("RecoParticlesNoLumiCal_subset", "AlephSelection::filterParticles(RecoParticles, _RecoParticles_tracks.index, ParticleID, _ParticleID_particle.index, AlephSelection::isAnalysisType())")
        df = df.Define("AnalysisParticles_subset", "AlephSelection::filterParticles(RecoParticlesNoLumiCal_subset, AlephSelection::notSoftNeutral(AlephSelection::kNeutralMinE))")
        df = df.Define("AnalysisParticles",         "AnalysisParticles_subset.particles")
        df = df.Define("AnalysisParticleID",        "AnalysisParticles_subset.pid")
        df = df.Define("_AnalysisParticles_tracks", "AnalysisParticles_subset.tracks")
        df = df.Define("AnalysisParticles_px", "ReconstructedParticle::get_px(AnalysisParticles)")
        df = df.Define("AnalysisParticles_py", "ReconstructedParticle::get_py(AnalysisParticles)")
        df = df.Define("AnalysisParticles_pz", "ReconstructedParticle::get_pz(AnalysisParticles)")
        df = df.Define("AnalysisParticles_e",  "ReconstructedParticle::get_e(AnalysisParticles)")


        # ------------------------------------------------------------
        # Two-jet clustering: each jet = one tau hemisphere
        # ------------------------------------------------------------
        df = df.Define("pjetc", "JetClusteringUtils::set_pseudoJets(AnalysisParticles_px, AnalysisParticles_py, AnalysisParticles_pz, AnalysisParticles_e)")
        df = df.Define("_jet", "JetClustering::clustering_ee_kt(2, 2, 1, 0)(pjetc)")
        df = df.Define("jets", "JetClusteringUtils::get_pseudoJets(_jet)")
        df = df.Define("_jetc", "JetClusteringUtils::get_constituents(_jet)")
        df = df.Define("jetc", "JetConstituentsUtils::build_constituents_cluster(AnalysisParticles, _jetc)")
        df = df.Define("jetConstituentsTypes", "AlephSelection::build_constituents_Types()(AnalysisParticleID, _jetc)")
        df = df.Define("event_njet", "JetConstituentsUtils::count_jets(jetc)")
        df = df.Filter("event_njet == 2", "2jets") #exactly two tau hemispheres

        # ------------------------------------------------------------
        # Particle-flow type flags, per jet per constituent
        # ------------------------------------------------------------
        df = df.Define("pfcand_isMu",         "AlephSelection::get_isType(jetConstituentsTypes,2)")
        df = df.Define("pfcand_isEl",         "AlephSelection::get_isType(jetConstituentsTypes,1)")
        df = df.Define("pfcand_isGamma",      "AlephSelection::get_isType(jetConstituentsTypes,4)")
        df = df.Define("pfcand_isChargedHad", "AlephSelection::get_isType(jetConstituentsTypes,0)")
        df = df.Define("pfcand_isNeutralHad", "AlephSelection::get_isType(jetConstituentsTypes,5)")
 

        # Charged multiplicity per jet == tau prong count.
        # AlephSelection::count_charged is NEW - see analyzer_ztautau.h
        df = df.Define("jet_nCharged", "AlephSelection::count_charged(pfcand_isChargedHad, pfcand_isMu, pfcand_isEl)")
 
        # Select one 1-prong hemisphere and one 3-prong hemisphere (either jet index)
        df = df.Filter("(jet_nCharged[0]==1 && jet_nCharged[1]==3) || (jet_nCharged[0]==3 && jet_nCharged[1]==1)", "1p3p_topology")

        """
        Prong count alone does not mean "pion": 1-prong is dominated by
        tau->e/mu nu-nu, and both prongs pick up extra pi0s from rho/a1
        decays. Of the two cuts below only (a) is applied; (b) is described
        but not applied. (a) does not cut on dE/dx directly, but ALEPH's
        electron identification (energy-flow type 1) uses dE/dx, so the veto
        is not fully independent of what the sample is later used to
        calibrate.
        (a) lepton veto: every charged constituent in the 3-prong jet must be a
        charged hadron, not e or mu (the 1-prong jet is not vetoed).
        """
        df = df.Define("jet_isHadronic", "AlephSelection::all_charged_are_hadronic(pfcand_isChargedHad, pfcand_isMu, pfcand_isEl)")
        # (b) no extra neutrals: total constituent count == charged count,
        # i.e. no accompanying pi0/photon - isolates pi-nu and 3pi-nu
        # specifically, excluding rho->pi pi0 and a1->3pi pi0.
        df = df.Define("j3", "jet_nCharged[0]==3 ? 0 : 1")
        df = df.Filter("jet_isHadronic[j3]", "3prong_hadronic")

        df = df.Define("j1", "jet_nCharged[0]==1 ? 0 : 1")

        # ------------------------------------------------------------
        # Vertex-proximity / TPC-crossing quality cuts
        # ------------------------------------------------------------
        # Raw impact parameters at the origin, as stage1's pfcand_d0 / pfcand_z0: the track state of each
        # constituent's own track, read through the particle -> track relation
        df = df.Define("TrackStateFlipped",f"AlephSelection::flipD0_copy( {coll['TrackState']} )")
        df = df.Define("TrackStateByRP", "AlephSelection::trackStatesByRPLink(TrackStateFlipped, Tracks, _AnalysisParticles_tracks)")
        df = df.Define("pfcand_d0",         "AlephSelection::get_constituent_trackParam(jetc, TrackStateByRP, 0)")
        df = df.Define("pfcand_z0",         "AlephSelection::get_constituent_trackParam(jetc, TrackStateByRP, 1)")
 
        # Per-jet worst-case |d0|, |z0| over its charged constituents - NEW helper.
        # Thresholds below (cm) are placeholders: tau flight cτ ≈ 87 μm, so
        # genuine 1-/3-prong decay products should sit very close to the
        # primary vertex; tune against your beamspot/IP resolution.
        df = df.Define("jet_max_absd0", "AlephSelection::max_abs_per_jet_charged(pfcand_d0, pfcand_isChargedHad, pfcand_isMu, pfcand_isEl)")
        df = df.Define("jet_max_absz0", "AlephSelection::max_abs_per_jet_charged(pfcand_z0, pfcand_isChargedHad, pfcand_isMu, pfcand_isEl)")

        
        df = df.Filter("jet_max_absd0[0]<2 && jet_max_absd0[1]<2 && "
               "jet_max_absz0[0]<10 && jet_max_absz0[1]<10", "IP_cuts")
        # Require every charged constituent to actually cross the TPC
        # (reuses the existing hit-counting helper from the qq analysis).
        # The hit counts and the track-quality branches below read each
        # constituent's own track (Tracks re-ordered through the relation).
        df = df.Define("TracksByRP", "AlephSelection::reindexByRPLink(Tracks, _AnalysisParticles_tracks)")
        df = df.Define("pfcand_nTrackHits_TPC", "AlephSelection::get_constituent_nTrackHits_TPC(jetc, TracksByRP, _Tracks_subdetectorHitNumbers)")
        df = df.Define("jet_min_TPChits_charged", "AlephSelection::min_TPChits_charged(pfcand_nTrackHits_TPC, pfcand_isChargedHad, pfcand_isMu, pfcand_isEl)")
        df = df.Filter("jet_min_TPChits_charged[0] > 0 && jet_min_TPChits_charged[1] > 0", "TPC_hits")

        # ------------------------------------------------------------
        # Pion kinematics (= the tau daughters)
        # ------------------------------------------------------------
        df = df.Define("pfcand_e",  "JetConstituentsUtils::get_e(jetc)")

        df = df.Filter(f"All(pfcand_e[0] > {MIN_CONSTITUENT_E}) && All(pfcand_e[1] > {MIN_CONSTITUENT_E})", "per_track_energy")

        # the two jets together above 10 % of the centre-of-mass energy, 2 x the LEP beam energy
        df = df.Filter("Sum(pfcand_e[0]) + Sum(pfcand_e[1]) > 0.1*2*EVT_Ebeam", "total_energy")
        df = df.Define("pfcand_p",  "JetConstituentsUtils::get_p(jetc)")
        df = df.Define("pfcand_px", "JetConstituentsUtils::get_px(jetc)")
        df = df.Define("pfcand_py", "JetConstituentsUtils::get_py(jetc)")
        df = df.Define("pfcand_pz", "JetConstituentsUtils::get_pz(jetc)")
        df = df.Define("pfcand_charge", "JetConstituentsUtils::get_charge(jetc)")

        # ------------------------------------------------------------
        # Track index, baseline track selection and the per-track ALEPH
        # variables of each constituent
        # ------------------------------------------------------------
        df = df.Define("pfcand_trackIdx", "AlephSelection::get_constituent_trackIdx(jetc, _AnalysisParticles_tracks)")
        # constituent k of jet j reads entry pfcand_trackIdx[j][k] of a per-track array; -1 is out of range
        for j in (0, 1):
            df = df.Define(f"_pfcand_take{j}", f"ROOT::VecOps::RVec<std::size_t>(pfcand_trackIdx[{j}].begin(), pfcand_trackIdx[{j}].end())")

        def per_constituent(per_track, elem, undefined):
            return (f"ROOT::VecOps::RVec<ROOT::VecOps::RVec<{elem}>>{{ROOT::VecOps::Take({per_track}, _pfcand_take0, {undefined}), "
                    f"ROOT::VecOps::Take({per_track}, _pfcand_take1, {undefined})}}")

        # stage1's baseline track selection: finite track state, positive-definite covariance, chi2/ndf <= 10,
        # AlephSelection::kTrackMinTPCHits TPC hits, |z0| <= AlephSelection::kTrackMaxAbsZ0
        df = df.Define("trkBaselineOrigIdx", "AlephSelection::select_tracks_baseline(Tracks, _Tracks_trackStates, _Tracks_subdetectorHitNumbers, "
                                             "AlephSelection::kTrackMinTPCHits, AlephSelection::kTrackMaxAbsZ0).origIdx")
        df = df.Define("trkBaseline", "ROOT::VecOps::RVec<int> f(Tracks.size(), 0); for (auto t : trkBaselineOrigIdx) f[t] = 1; return f;")
        df = df.Define("pfcand_trkBaseline", per_constituent("trkBaseline", "int", "-1"))

        # ------------------------------------------------------------
        # dE/dx with ALPHA's particle hypotheses (converter collections, as
        # stage1's trk_dEdx_* and the Ks example). These branches are the
        # calibration TARGET - do not filter on them above, or the sample
        # purity becomes circular.
        # ------------------------------------------------------------
        self.dedx_branches = []
        # (branch infix; collection; parameter count; empty collection allowed)
        dets = [("wires", "dEdxWiresPID", "kNParDedx", "false"),
                ("pads", "dEdxPadsPID", "kNParDedx", "false")]
        if not self.ana_args.doData:
            dets.append(("wiresQDEDXM", "dEdxWiresQDEDXMPID", "kNParQDEDXM", "true"))
        for name, dcoll, npar, empty_ok in dets:
            df = df.Define(f"trkDedx_{name}", f'{AV}::trackDedx({dcoll}, _{dcoll}_parameters, Tracks.size(), '
                                              f'{AV}::{npar}, "{dcoll}", {empty_ok})')
            members = [("value", "value")] + ([("nSamples", "nSamples")] if name != "wiresQDEDXM" else []) + \
                      [(f"{q}_{h}", f"{q}[{k}]") for k, h in enumerate(DEDX_HYPOS) for q in ("expected", "sigma", "pull")]
            for b, m in members:
                self.dedx_branches.append(f"pfcand_dEdx_{name}_{b}")
                df = df.Define(self.dedx_branches[-1], per_constituent(f"trkDedx_{name}.{m}", "float", f"{AV}::kNotAvailable"))

        # ------------------------------------------------------------
        # Invariant masses: tau visible mass (jet) and charged-only
        # ("daughters") mass. Both are NEW helpers.
        # ------------------------------------------------------------
        df = df.Define("jet_p4", "JetConstituentsUtils::compute_tlv_jets(jets)")
        df = df.Define("tau_visible_mass", "AlephSelection::get_masses(jet_p4)")
        df = df.Define("tau_charged_mass", "AlephSelection::get_charged_invmass(pfcand_px, pfcand_py, pfcand_pz, pfcand_e, "
                                             "pfcand_isChargedHad, pfcand_isMu, pfcand_isEl)")
        df = df.Define("tau3_visible_mass", "tau_visible_mass[j3]")
        df = df.Define("tau3_charged_mass", "tau_charged_mass[j3]")
 
        # Track-quality bookkeeping (kept from the qq treemaker)
        df = df.Define("pfcand_trackChi2",     "AlephSelection::get_constituent_trackChi2(jetc, TracksByRP)")
        df = df.Define("pfcand_trackNdof",     "AlephSelection::get_constituent_trackNdof(jetc, TracksByRP)")
        df = df.Define("pfcand_trackChi2Norm", "AlephSelection::get_constituent_trackChi2Norm(jetc, TracksByRP)")
        df = df.Define("pfcand_nTrackHits_VDET", "AlephSelection::get_constituent_nTrackHits_VDET(jetc, TracksByRP, _Tracks_subdetectorHitNumbers)")
        df = df.Define("pfcand_nTrackHits_ITC",  "AlephSelection::get_constituent_nTrackHits_ITC(jetc, TracksByRP, _Tracks_subdetectorHitNumbers)")

        if not self.ana_args.doData:
            df = df.Define("pfcand_mc_pdg", per_constituent("trkBestMatch.pdg", "int", f"{AV}::kNotAvailableInt"))
            # float, the index type of the truth helpers below
            df = df.Define("pfcand_mc_index", per_constituent("ROOT::VecOps::RVec<float>(trkBestMatch.idx.begin(), trkBestMatch.idx.end())",
                                                              "float", f"{AV}::kNotAvailable"))
            df = df.Define("pfcand_mc_momentum", per_constituent("trkBestMatch.p", "float", f"{AV}::kNotAvailable"))

        # slice every per-jet / per-constituent branch down to just that hemisphere
        self.slice_branches = []
        sliced = ["pfcand_isMu", "pfcand_isEl", "pfcand_isGamma", "pfcand_isChargedHad", "pfcand_isNeutralHad",
                  "pfcand_e", "pfcand_p", "pfcand_px", "pfcand_py", "pfcand_pz", "pfcand_charge", "pfcand_d0", "pfcand_z0",
                  "pfcand_nTrackHits_TPC", "pfcand_trackIdx", "pfcand_trkBaseline"] + self.dedx_branches
        if not self.ana_args.doData:
            sliced += ["pfcand_mc_pdg", "pfcand_mc_index", "pfcand_mc_momentum"]
        for b in sliced:
            for h in ("3", "1"):
                self.slice_branches.append(b.replace("pfcand_", f"pfcand{h}_"))
                df = df.Define(self.slice_branches[-1], f"{b}[j{h}]")

        if not self.ana_args.doData:
            """
            df = df.Define("jet_truth_decay_mode",
                "AlephSelection::get_jet_truth_decay_mode(pfcand_mc_index, MCParticles, "
                "_MCParticles_parents.index, _MCParticles_daughters.index)")

            df = df.Define("jet_truth_is_single_tau",
                "AlephSelection::jet_is_single_tau_matched(pfcand_mc_index, MCParticles, "
                "_MCParticles_parents.index)")

            df = df.Define("jet3_truth_decay_mode",    "jet_truth_decay_mode[j3]")
            df = df.Define("jet3_truth_is_single_tau", "jet_truth_is_single_tau[j3]")
            """

            df = df.Define("jet3_mc_index", "pfcand_mc_index[j3]")
            df = df.Define("MC_vertex_x", "MCParticle::get_vertex_x(MCParticles)")
            df = df.Define("MC_vertex_y", "MCParticle::get_vertex_y(MCParticles)")
            df = df.Define("MC_vertex_z", "MCParticle::get_vertex_z(MCParticles)")
            df = df.Define("jet3_same_vertex",
            "AlephSelection::jet_same_vertex(jet3_mc_index, MC_vertex_x, MC_vertex_y, MC_vertex_z, 1e-6f)")
            df = df.Define("truth_tau3pi_mass",
            "AlephSelection::truth_tau_mass_3pi_nu(jet3_mc_index, MCParticles, "
            "MC_vertex_x, MC_vertex_y, MC_vertex_z, 1e-3f)")

        return df
        

    def output(self):
        branchList = [
            "event_number",
            "run_number",
            "event_class",
            "EVT_Ebeam",

            "jet_nCharged",
            "jet_max_absd0",
            "jet_max_absz0",
            "jet_min_TPChits_charged",
            "tau_visible_mass",
            "tau_charged_mass",
            "tau3_visible_mass",
            "tau3_charged_mass",

            "pfcand_isMu", "pfcand_isEl", "pfcand_isChargedHad", "pfcand_isGamma", "pfcand_isNeutralHad",
            "pfcand_e", "pfcand_p", "pfcand_px", "pfcand_py", "pfcand_pz", "pfcand_charge",
            "pfcand_d0", "pfcand_z0",
            "pfcand_nTrackHits_VDET", "pfcand_nTrackHits_ITC", "pfcand_nTrackHits_TPC",
            "pfcand_trackChi2", "pfcand_trackNdof", "pfcand_trackChi2Norm",
            "pfcand_trackIdx", "pfcand_trkBaseline",
        ] + self.dedx_branches + self.slice_branches

        if not self.ana_args.doData:
            branchList += [
                "MC_pdg", "MC_status",
                "MC_momentum_x", "MC_momentum_y", "MC_momentum_z",

                "pfcand_mc_pdg", "pfcand_mc_index", "pfcand_mc_momentum",

                "jet3_mc_index",
                "MC_vertex_x", "MC_vertex_y", "MC_vertex_z",
                "jet3_same_vertex", "truth_tau3pi_mass",
            ]

        return branchList