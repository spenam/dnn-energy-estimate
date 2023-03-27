import ROOT
import km3ant as ka
import numpy as np

#####################################################
# Defining data sets
# ------------------
#
# In km3ant, datasets are declared in a dictionnary.
# Each dataset is defined as pair :
#
# ``<dataset label>:{"fileng data sets
# ------------------
#
# In km3ant, datasets are declared in a dictionnary.
# Each dataset is defined as pair :
#
# ``<dataset label>:{"filename":<dstfilepath>}``
#

dst_files_directory = "/sps/km3net/users/alflazo/dstProd/v7.1_v7.2_jsh/"
# friend_files_directory = "/sps/km3net/users/spenamar/oscillations/cuts_analysis/friendPIDs"
friend_files_directory = "/sps/km3net/users/spenamar/oscillations/PIDfilesAlfonso/"
friend_files_directory2 = "/sps/km3net/users/spenamar/hackathon/tf/training_workflow/model_pipeline/results/v7.10"
# For Alfonso's PID swim files v7.2
dataset = {
    "muons v7.2": {
        "filename": dst_files_directory
        + "/v7.2.mc.mupage_tuned.sirene.jorcarec.jsh.aanet.dst_merged.root"
    },
    "anu e CC high v7.2": {
        "filename": dst_files_directory
        + "/mcv7.2.gsg_anti-elec-CC_100-500GeV.sirene.jorcarec.jsh.aanet.dst_merged.root"
    },
    "anu e CC low v7.2": {
        "filename": dst_files_directory
        + "/mcv7.2.gsg_anti-elec-CC_1-100GeV.km3sim.jorcarec.jsh.aanet.dst_merged.root"
    },
    "anu e CC higher v7.2": {
        "filename": dst_files_directory
        + "/mcv7.2.gsg_anti-elec-CCHEDIS_500-10000GeV.sirene.jorcarec.jsh.aanet.dst_merged.root"
    },
    "anu mu CC high v7.2": {
        "filename": dst_files_directory
        + "/mcv7.2.gsg_anti-muon-CC_100-500GeV.km3sim.jorcarec.jsh.aanet.dst_merged.root"
    },
    "anu mu CC low v7.2": {
        "filename": dst_files_directory
        + "/mcv7.2.gsg_anti-muon-CC_1-100GeV.km3sim.jorcarec.jsh.aanet.dst_merged.root"
    },
    "anu mu CC higher v7.2": {
        "filename": dst_files_directory
        + "/mcv7.2.gsg_anti-muon-CCHEDIS_500-10000GeV.sirene.jorcarec.jsh.aanet.dst_merged.root"
    },
    "anu mu NC high v7.2": {
        "filename": dst_files_directory
        + "/mcv7.2.gsg_anti-muon-NC_100-500GeV.sirene.jorcarec.jsh.aanet.dst_merged.root"
    },
    "anu mu NC low v7.2": {
        "filename": dst_files_directory
        + "/mcv7.2.gsg_anti-muon-NC_1-100GeV.km3sim.jorcarec.jsh.aanet.dst_merged.root"
    },
    "anu mu NC higher v7.2": {
        "filename": dst_files_directory
        + "/mcv7.2.gsg_anti-muon-NCHEDIS_500-10000GeV.sirene.jorcarec.jsh.aanet.dst_merged.root"
    },
    "anu tau CC high v7.2": {
        "filename": dst_files_directory
        + "/mcv7.2.gsg_anti-tau-CC_100-500GeV.sirene.jorcarec.jsh.aanet.dst_merged.root"
    },
    "anu tau CC low v7.2": {
        "filename": dst_files_directory
        + "/mcv7.2.gsg_anti-tau-CC_3-100GeV.km3sim.jorcarec.jsh.aanet.dst_merged.root"
    },
    "nu e CC high v7.2": {
        "filename": dst_files_directory
        + "/mcv7.2.gsg_elec-CC_100-500GeV.sirene.jorcarec.jsh.aanet.dst_merged.root"
    },
    "nu e CC low v7.2": {
        "filename": dst_files_directory
        + "/mcv7.2.gsg_elec-CC_1-100GeV.km3sim.jorcarec.jsh.aanet.dst_merged.root"
    },
    "nu e CC higher v7.2": {
        "filename": dst_files_directory
        + "/mcv7.2.gsg_elec-CCHEDIS_500-10000GeV.sirene.jorcarec.jsh.aanet.dst_merged.root"
    },
    "nu mu CC high v7.2": {
        "filename": dst_files_directory
        + "/mcv7.2.gsg_muon-CC_100-500GeV.km3sim.jorcarec.jsh.aanet.dst_merged.root"
    },
    "nu mu CC low v7.2": {
        "filename": dst_files_directory
        + "/mcv7.2.gsg_muon-CC_1-100GeV.km3sim.jorcarec.jsh.aanet.dst_merged.root"
    },
    "nu mu CC higher v7.2": {
        "filename": dst_files_directory
        + "/mcv7.2.gsg_muon-CCHEDIS_500-10000GeV.sirene.jorcarec.jsh.aanet.dst_merged.root"
    },
    "nu mu NC high v7.2": {
        "filename": dst_files_directory
        + "/mcv7.2.gsg_muon-NC_100-500GeV.sirene.jorcarec.jsh.aanet.dst_merged.root"
    },
    "nu mu NC low v7.2": {
        "filename": dst_files_directory
        + "/mcv7.2.gsg_muon-NC_1-100GeV.km3sim.jorcarec.jsh.aanet.dst_merged.root"
    },
    "nu mu NC higher v7.2": {
        "filename": dst_files_directory
        + "/mcv7.2.gsg_muon-NCHEDIS_500-10000GeV.sirene.jorcarec.jsh.aanet.dst_merged.root"
    },
    "nu tau CC high v7.2": {
        "filename": dst_files_directory
        + "/mcv7.2.gsg_tau-CC_100-500GeV.sirene.jorcarec.jsh.aanet.dst_merged.root"
    },
    "nu tau CC low v7.2": {
        "filename": dst_files_directory
        + "/mcv7.2.gsg_tau-CC_3-100GeV.km3sim.jorcarec.jsh.aanet.dst_merged.root"
    },
    "muons7X8X v7.1": {
        "filename": dst_files_directory
        + "/v7.1.mc.mupage_tuned.sirene.jorcarec.jsh.aanet.7X_8X_dst_merged.root"
    },
    "muons9X10X v7.1": {
        "filename": dst_files_directory
        + "/v7.1.mc.mupage_tuned.sirene.jorcarec.jsh.aanet.9X_10X_dst_merged.root"
    },
    "anu e CC high v7.1": {
        "filename": dst_files_directory
        + "/mcv7.1.gsg_anti-elec-CC_100-500GeV.sirene.jorcarec.jsh.aanet.dst_merged.root"
    },
    "anu e CC low v7.1": {
        "filename": dst_files_directory
        + "/mcv7.1.gsg_anti-elec-CC_1-100GeV.km3sim.jorcarec.jsh.aanet.dst_merged.root"
    },
    "anu e CC higher v7.1": {
        "filename": dst_files_directory
        + "/mcv7.1.gsg_anti-elec-CCHEDIS_500-10000GeV.sirene.jorcarec.jsh.aanet.dst_merged.root"
    },
    # 'anu mu CC high v7.1'     : {'filename':dst_files_directory+'/mcv7.1.gsg_anti-muon-CC_100-500GeV.sirene.jorcarec.jsh.aanet.dst_merged.root'},
    #    'anu mu CC low v7.1'      : {'filename':dst_files_directory+'/mcv7.1.gsg_anti-muon-CC_1-100GeV.km3sim.jorcarec.jsh.aanet.dst_merged.root'},
    "anu mu CC higher v7.1": {
        "filename": dst_files_directory
        + "/mcv7.1.gsg_anti-muon-CCHEDIS_500-10000GeV.sirene.jorcarec.jsh.aanet.dst_merged.root"
    },
    "anu mu NC high v7.1": {
        "filename": dst_files_directory
        + "/mcv7.1.gsg_anti-muon-NC_100-500GeV.sirene.jorcarec.jsh.aanet.dst_merged.root"
    },
    "anu mu NC low v7.1": {
        "filename": dst_files_directory
        + "/mcv7.1.gsg_anti-muon-NC_1-100GeV.km3sim.jorcarec.jsh.aanet.dst_merged.root"
    },
    "anu mu NC higher v7.1": {
        "filename": dst_files_directory
        + "/mcv7.1.gsg_anti-muon-NCHEDIS_500-10000GeV.sirene.jorcarec.jsh.aanet.dst_merged.root"
    },
    #    'anu tau CC high v7.1'    : {'filename':dst_files_directory+'/mcv7.1.gsg_anti-tau-CC_100-500GeV.sirene.jorcarec.jsh.aanet.dst_merged.root'},
    #    'anu tau CC low v7.1'     : {'filename':dst_files_directory+'/mcv7.1.gsg_anti-tau-CC_3-100GeV.km3sim.jorcarec.jsh.aanet.dst_merged.root'},
    "nu e CC high v7.1": {
        "filename": dst_files_directory
        + "/mcv7.1.gsg_elec-CC_100-500GeV.sirene.jorcarec.jsh.aanet.dst_merged.root"
    },
    "nu e CC low v7.1": {
        "filename": dst_files_directory
        + "/mcv7.1.gsg_elec-CC_1-100GeV.km3sim.jorcarec.jsh.aanet.dst_merged.root"
    },
    "nu e CC higher v7.1": {
        "filename": dst_files_directory
        + "/mcv7.1.gsg_elec-CCHEDIS_500-10000GeV.sirene.jorcarec.jsh.aanet.dst_merged.root"
    },
    # ` 'nu mu CC high v7.1'     : {'filename':dst_files_directory+'/mcv7.1.gsg_muon-CC_100-500GeV.sirene.jorcarec.jsh.aanet.dst_merged.root'},
    #    'nu mu CC low v7.1'      : {'filename':dst_files_directory+'/mcv7.1.gsg_muon-CC_1-100GeV.km3sim.jorcarec.jsh.aanet.dst_merged.root'},
    "nu mu CC higher v7.1": {
        "filename": dst_files_directory
        + "/mcv7.1.gsg_muon-CCHEDIS_500-10000GeV.sirene.jorcarec.jsh.aanet.dst_merged.root"
    },
    "nu mu NC high v7.1": {
        "filename": dst_files_directory
        + "/mcv7.1.gsg_muon-NC_100-500GeV.sirene.jorcarec.jsh.aanet.dst_merged.root"
    },
    "nu mu NC low v7.1": {
        "filename": dst_files_directory
        + "/mcv7.1.gsg_muon-NC_1-100GeV.km3sim.jorcarec.jsh.aanet.dst_merged.root"
    },
    "nu mu NC higher v7.1": {
        "filename": dst_files_directory
        + "/mcv7.1.gsg_muon-NCHEDIS_500-10000GeV.sirene.jorcarec.jsh.aanet.dst_merged.root"
    },
    #    'nu tau CC high v7.1'    : {'filename':dst_files_directory+'/mcv7.1.gsg_tau-CC_100-500GeV.sirene.jorcarec.jsh.aanet.dst_merged.root'},
    #    'nu tau CC low v7.1'     : {'filename':dst_files_directory+'/mcv7.1.gsg_tau-CC_3-100GeV.km3sim.jorcarec.jsh.aanet.dst_merged.root'},
}
dataset_data = {
    "data 7X": {
        "filename": dst_files_directory
        + "/datav7.1.jorcarec.jsh.aanet.7X_dst_merged.root"
    },
    "data 80": {
        "filename": dst_files_directory
        + "/datav7.1.jorcarec.jsh.aanet.80_dst_merged.root"
    },
    "data 85": {
        "filename": dst_files_directory
        + "/datav7.1.jorcarec.jsh.aanet.85_dst_merged.root"
    },
    "data 90": {
        "filename": dst_files_directory
        + "/datav7.1.jorcarec.jsh.aanet.90_dst_merged.root"
    },
    "data 95": {
        "filename": dst_files_directory
        + "/datav7.1.jorcarec.jsh.aanet.95_dst_merged.root"
    },
    "data 100X": {
        "filename": dst_files_directory
        + "/datav7.1.jorcarec.jsh.aanet.100X_dst_merged.root"
    },
    "data 110X": {
        "filename": dst_files_directory
        + "/datav7.1.jorcarec.jsh.aanet.110X_dst_merged.root"
    },
}

for key, items in dataset.items():
    items["pipename"] = key  # Don't remember why but that's important
    fname = items["filename"].split("/")[
        -1
    ]  # Get the DST filename to convert it into GNN file name
    fname = "outputTree_PID_" + fname + ".root"
    items["filename_friend"] = friend_files_directory + \
        "/" + fname.split("/")[-1]
    # items['friend_trees'] = ['T',items['filename_friend']+':events']  # Tells the pump to load the tree "T" and "<path_to_the_file>:gnn", in addition to the E tree name":<dstfilepath>}``
    # items['friend_trees'] = ['T',items['filename_friend']+':tree']  # Tells the pump to load the tree "T" and "<path_to_the_file>:gnn", in addition to the E tree name":<dstfilepath>}``
    fname = (
        items["filename"].split("/")[-1].split(".root")[0]
    )  # Get the DST filename to convert it into GNN file name
    fname = fname + "_DNN.root"
    items["filename_friend2"] = friend_files_directory2 + \
        "/" + fname.split("/")[-1]
    # items['friend_trees'] = ['T',items['filename_friend']+':events']  # Tells the pump to load the tree "T" and "<path_to_the_file>:gnn", in addition to the E tree name":<dstfilepath>}``
    items["friend_trees"] = [
        "T",
        items["filename_friend"] + ":tree",
        items["filename_friend2"] + ":DNN",
    ]  # Tells the pump to load the tree "T" and "<path_to_the_file>:gnn", in addition to the E tree name":<dstfilepath>}``
for key, items in dataset_data.items():
    items["pipename"] = key  # Don't remember why but that's important
    fname = items["filename"].split("/")[
        -1
    ]  # Get the DST filename to convert it into GNN file name
    fname = "outputTree_PID_" + fname + ".root"
    items["filename_friend"] = friend_files_directory + \
        "/" + fname.split("/")[-1]
    # items['friend_trees'] = ['T',items['filename_friend']+':events']  # Tells the pump to load the tree "T" and "<path_to_the_file>:gnn", in addition to the E tree name":<dstfilepath>}``
    # items['friend_trees'] = ['T',items['filename_friend']+':tree']  # Tells the pump to load the tree "T" and "<path_to_the_file>:gnn", in addition to the E tree name":<dstfilepath>}``
    fname = (
        items["filename"].split("/")[-1].split(".root")[0]
    )  # Get the DST filename to convert it into GNN file name
    fname = fname + "_DNN.root"
    items["filename_friend2"] = friend_files_directory2 + \
        "/" + fname.split("/")[-1]
    # items['friend_trees'] = ['T',items['filename_friend']+':events']  # Tells the pump to load the tree "T" and "<path_to_the_file>:gnn", in addition to the E tree name":<dstfilepath>}``
    items["friend_trees"] = [
        "T",
        items["filename_friend"] + ":tree",
        items["filename_friend2"] + ":DNN",
    ]  # Tells the pump to load the tree "T" and "<path_to_the_file>:gnn", in addition to the E tree name":<dstfilepath>}``
print(dataset)


#####################################################
# Setting up the pipe
# -------------------
#
# In this part, we will create the pipe layout
#
# Creating the pipe manager
# ~~~~~~~~~~~~~~~~~~~~~~~~~
#
# All the modules will be registered to this module. It is also the
# one taking care of creating the pump.  It takes as argument the
# dictionnary containing the datasets.  Here, we also decide to set
# the step_size, i.e. the number of events in each data blob.

manager = ka.pipeManager(dataset)
manager_data = ka.pipeManager(dataset_data)

manager.append_module(
    "ant_pump",
    ka.AntPump,
    kwargs={"step_size": 1000000},
    dataset_fields={
        "filename": "filename",
        "friend_trees": "friend_trees",
        "pipename": "pipename",
    },
)
manager_data.append_module(
    "ant_pump",
    ka.AntPump,
    kwargs={"step_size": 1000000},
    dataset_fields={
        "filename": "filename",
        "friend_trees": "friend_trees",
        "pipename": "pipename",
    },
)
#####################################################
# Exposure handler
# ~~~~~~~~~~~~~~~~~~~~~~
#
# The exposure handler is a basic module that will keep track of the
# exposure time.  This is a mandatory module because per default
# histogram are filled in rates.
#
# The name of the module has to be specified, this way you can imagine
# having multiple exposure module in paralel, but that's something we
# will discuss in more advancede examples.

manager.append_module("exposure", ka.exposureHandler)
manager_data.append_module("exposure", ka.exposureHandler)
#####################################################
# Event weighter
# ~~~~~~~~~~~~~~~~~~~~~~
#
# As the name might suggest, this module will compute the weight of
# each event.  In this example, as we use data, the weight should be
# set to 1 for all events, and that's what ``{'cst_weight':1}`` does.
# Weighting class from km3ant
# weighter = ka.weighter_OscProb_km3flux()

# set the weighing scheme :
for key, value in dataset.items():
    value["ext_weight"] = ""

    if key.find("data") != -1:  # Data, will use cst weight of 1
        pass
    # Atm. muons, use pre-computed weight from DST (livetime_sim/livetime_DAQ, per run)
    else:
        value["ext_weight"] = "T.sum_mc_evt.weight"

for key, value in dataset_data.items():
    value["ext_weight"] = ""

    if key.find("data") != -1:  # Data, will use cst weight of 1
        pass
    # Atm. muons, use pre-computed weight from DST (livetime_sim/livetime_DAQ, per run)
    else:
        value["ext_weight"] = "T.sum_mc_evt.weight"


manager.append_module(
    "weighter",
    ka.eventWeighter,
    dataset_fields={"ext_weight": "ext_weight"},
    kwargs={"cst_weight": 1},
)
manager_data.append_module(
    "weighter",
    ka.eventWeighter,
    dataset_fields={"ext_weight": "ext_weight"},
    kwargs={"cst_weight": 1},
)
#####################################################
# Adding some cuts
# ~~~~~~~~~~~~~~~~~~~~~
#
# Cut are handled through masks. The idea is to compute
# a vector of boolean that represant which events
# survive the cuts.
#
# Here, we apply two cuts :
#
# ``E.trks.lik[:,0]`` point to first track likelihood
# and is required to be above 40.
#
# ``E.trks.E[:,0]`` point to first track energy
# and is required to be below 100 GeV.

min_trig_hits = 15
min_lik = 40
min_dir_z = 0

without = []

# Define some aliases to compute new variables on the fly while laoding the file
# Note that for now, aliases are required only for the friend tree T in dst files


cuts = [
    ka.lin_cut("E.trks.lik[:,0]", ">", min_lik),
    ka.lin_cut("E.trks.dir.z[:,0]", ">", min_dir_z),
    ka.lin_cut("T.sum_trig_hits.nhits", ">", min_trig_hits),
]
pure_track = cuts + [
    ka.lin_cut("tree.trackscore", ">", 0.85),
    ka.lin_cut("tree.muonscore", "<", 1e-4),
]
pure_shower = cuts + [
    ka.lin_cut("tree.trackscore", "<=", 0.85),
    ka.lin_cut("tree.muonscore", "<", 1e-4),
]
contaminated_track = cuts + [
    ka.lin_cut("tree.trackscore", ">", 0.85),
    ka.lin_cut("tree.muonscore", "<", 2e-3),
    ka.lin_cut("tree.muonscore", ">", 1e-4),
]
contaminated_shower = cuts + [
    ka.lin_cut("tree.trackscore", "<=", 0.85),
    ka.lin_cut("tree.muonscore", "<", 2e-3),
    ka.lin_cut("tree.muonscore", ">", 1e-4),
]


manager.append_module("without", ka.cutsHandler, kwargs={"cuts": without})
manager.append_module("my_simple_cuts", ka.cutsHandler, kwargs={"cuts": cuts})
manager.append_module("pure_track", ka.cutsHandler,
                      kwargs={"cuts": pure_track})
manager.append_module("pure_shower", ka.cutsHandler,
                      kwargs={"cuts": pure_shower})
manager.append_module(
    "contaminated_track", ka.cutsHandler, kwargs={"cuts": contaminated_track}
)
manager.append_module(
    "contaminated_shower", ka.cutsHandler, kwargs={"cuts": contaminated_shower}
)
manager_data.append_module("without", ka.cutsHandler, kwargs={"cuts": without})
manager_data.append_module(
    "my_simple_cuts", ka.cutsHandler, kwargs={"cuts": cuts})
manager_data.append_module(
    "pure_track", ka.cutsHandler, kwargs={"cuts": pure_track})
manager_data.append_module(
    "pure_shower", ka.cutsHandler, kwargs={"cuts": pure_shower})
manager_data.append_module(
    "contaminated_track", ka.cutsHandler, kwargs={"cuts": contaminated_track}
)
manager_data.append_module(
    "contaminated_shower", ka.cutsHandler, kwargs={"cuts": contaminated_shower}
)


#####################################################
# Adding some histograms
# ~~~~~~~~~~~~~~~~~~~~~~
#
# Now that we have some cuts, let's fill histogram, with and without the cuts.
#
# First, we are declaring a list of kc.histogram objects.  This object
# expects a name, the variable in string format and the bins.  The 2
# first histograms are 1D histograms, for cos theta and energy. Note
# that energy will have log binning. The third one shows how to do 2D
# histograms.
#
# Then, we declare 2 histogram handler that we start with the same
# list of histogram.  The first one doesn't have any mask declared, it
# will use every events.  The second one receive as ``mask_name``
# argument the name of cut handler declared previously. It will use
# only events surviving these cuts.

E_bins = 101
n_bins = 200
n_bins = 100

# real_hists = [
#        ka.histogram("cos_theta", "E.mc_trks.dir.z[:,0]", np.linspace(-1,1,41)),
#        ka.histogram("energy_log", "E.mc_trks.E[:,0]", 10**np.linspace(0,2,E_bins)),#41))
# ]

mc_hists = [
    ka.histogram("True_E", "E.mc_trks.E[:,0]", np.logspace(-1, 4, n_bins)),
    ka.histogram(
        "trueEnergy_vs_JG_E",
        ["E.mc_trks.E[:,0]", "E.trks.E[:,0]"],
        [10 ** np.linspace(0, 4, 41), 10 ** np.linspace(0, 4, 41)],
    ),
    ka.histogram(
        "trueEnergy_vs_JG_Len",
        ["E.mc_trks.E[:,0]", "E.trks.len[:,0]*0.25"],
        [10 ** np.linspace(0, 4, 41), 10 ** np.linspace(0, 4, 41)],
    ),
    ka.histogram(
        "trueEnergy_vs_JS_E",
        ["E.mc_trks.E[:,0]", "E.trks.E[:,1]"],
        [10 ** np.linspace(0, 4, 41), 10 ** np.linspace(0, 4, 41)],
    ),
    ka.histogram(
        "trueEnergy_vs_DNN_Eest",
        ["E.mc_trks.E[:,0]", "DNN.energy_recoDNN"],
        [10 ** np.linspace(0, 4, 41), 10 ** np.linspace(0, 4, 41)],
    ),
]
data_hists = [
    ka.histogram("track_score", "tree.trackscore", np.linspace(0, 1, n_bins)),
    ka.histogram("muon_score", "tree.muonscore", np.logspace(-7, 0, n_bins)),
    ka.histogram("JG_E", "E.trks.E[:,0]", np.logspace(-1, 4, n_bins)),
    ka.histogram("JG_Len", "E.trks.len[:,0]*0.25", np.logspace(-1, 4, n_bins)),
    ka.histogram("JS_E", "E.trks.E[:,1]", np.logspace(-1, 4, n_bins)),
    ka.histogram("DNN_Eest", "DNN.energy_recoDNN", np.logspace(-1, 4, n_bins)),
]


manager.append_module(
    "mc_hists_without_cuts",
    ka.histogramHandler,
    kwargs={"histograms": mc_hists, "rate_hists": True},
)


manager.append_module(
    "mc_hists_with_simple_cuts",
    ka.histogramHandler,
    kwargs={"mask_name": "my_simple_cuts",
            "histograms": mc_hists, "rate_hists": True},
)

manager.append_module(
    "mc_hists_pure_track",
    ka.histogramHandler,
    kwargs={"mask_name": "pure_track",
            "histograms": mc_hists, "rate_hists": True},
)
manager.append_module(
    "mc_hists_pure_shower",
    ka.histogramHandler,
    kwargs={"mask_name": "pure_shower",
            "histograms": mc_hists, "rate_hists": True},
)
manager.append_module(
    "mc_hists_contaminated_track",
    ka.histogramHandler,
    kwargs={
        "mask_name": "contaminated_track",
        "histograms": mc_hists,
        "rate_hists": True,
    },
)
manager.append_module(
    "mc_hists_contaminated_shower",
    ka.histogramHandler,
    kwargs={
        "mask_name": "contaminated_shower",
        "histograms": mc_hists,
        "rate_hists": True,
    },
)

manager_data.append_module(
    "data_hists_without_cuts",
    ka.histogramHandler,
    kwargs={"histograms": data_hists, "rate_hists": True},
)


manager_data.append_module(
    "data_hists_with_simple_cuts",
    ka.histogramHandler,
    kwargs={
        "mask_name": "my_simple_cuts",
        "histograms": data_hists,
        "rate_hists": True,
    },
)

manager_data.append_module(
    "data_hists_pure_track",
    ka.histogramHandler,
    kwargs={"mask_name": "pure_track",
            "histograms": data_hists, "rate_hists": True},
)
manager_data.append_module(
    "data_hists_pure_shower",
    ka.histogramHandler,
    kwargs={"mask_name": "pure_shower",
            "histograms": data_hists, "rate_hists": True},
)
manager_data.append_module(
    "data_hists_contaminated_track",
    ka.histogramHandler,
    kwargs={
        "mask_name": "contaminated_track",
        "histograms": data_hists,
        "rate_hists": True,
    },
)
manager_data.append_module(
    "data_hists_contaminated_shower",
    ka.histogramHandler,
    kwargs={
        "mask_name": "contaminated_shower",
        "histograms": data_hists,
        "rate_hists": True,
    },
)


#####################################################
# Run the pipeline
# ~~~~~~~~~~~~~~~~~~~~~~
#
# Finally, we have defined what we want to do, we can run the analysis.

manager.run()

#####################################################
# Saving results
# --------------
#
# For now, no automatic way to export the results exists.  Here is a
# small example on how exporting the results to a ROOT file, and
# displaying some info about the cuts.  ``manager.results`` is a
# dictionnary, where the key is the module name, and the value the
# module itself.
#
# In this loop, we are looping over the modules, and applying
# different action in function of what they are. Histogram handler and
# exposure handler are having a ``exportToRoot`` function, that take a
# TDirectory as argument.


# outfile = ROOT.TFile('mc_hists.root', 'RECREATE')
# outfile = ROOT.TFile('mc_hists_v7.2.root', 'RECREATE')
outfile = ROOT.TFile("mc_hists.root", "RECREATE")

sub_menu = [
    [
        [
            "anu e CC low v7.2",
            "+",
            "anu e CC high v7.2",
            "+",
            "anu e CC higher v7.2",
            "+",
            "nu e CC low v7.2",
            "+",
            "nu e CC high v7.2",
            "+",
            "nu e CC higher v7.2",
        ],
        "nu e CC tot v7.2",
    ],
    [
        [
            "anu mu CC low v7.2",
            "+",
            "anu mu CC high v7.2",
            "+",
            "anu mu CC higher v7.2",
            "+",
            "nu mu CC low v7.2",
            "+",
            "nu mu CC high v7.2",
            "+",
            "nu mu CC higher v7.2",
        ],
        "nu mu CC tot v7.2",
    ],
    [
        [
            "anu mu NC low v7.2",
            "+",
            "anu mu NC high v7.2",
            "+",
            "anu mu NC higher v7.2",
            "+",
            "nu mu NC low v7.2",
            "+",
            "nu mu NC high v7.2",
            "+",
            "nu mu NC higher v7.2",
        ],
        "nu mu NC tot v7.2",
    ],
    [
        [
            "anu tau CC low v7.2",
            "+",
            "anu tau CC high v7.2",
            "+",
            "nu tau CC low v7.2",
            "+",
            "nu tau CC high v7.2",
        ],
        "nu tau CC tot v7.2",
    ],
    [
        [
            "nu e CC tot v7.2",
            "+",
            "nu mu CC tot v7.2",
            "+",
            "nu mu NC tot v7.2",
            "+",
            "nu tau CC tot v7.2",
        ],
        "nu total v7.2",
    ],
    [
        [
            "nu total v7.2",
            "+",
            "muons v7.2",
        ],
        "mc total v7.2",
    ],
    [
        [
            "anu e CC low v7.1",
            "+",
            "anu e CC high v7.1",
            "+",
            "anu e CC higher v7.1",
            "+",
            "nu e CC low v7.1",
            "+",
            "nu e CC high v7.1",
            "+",
            "nu e CC higher v7.1",
        ],
        "nu e CC tot v7.1",
    ],
    [["anu mu CC higher v7.1", "+", "nu mu CC higher v7.1"], "nu mu CC tot v7.1"],
    [
        [
            "anu mu NC low v7.1",
            "+",
            "anu mu NC high v7.1",
            "+",
            "anu mu NC higher v7.1",
            "+",
            "nu mu NC low v7.1",
            "+",
            "nu mu NC high v7.1",
            "+",
            "nu mu NC higher v7.1",
        ],
        "nu mu NC tot v7.1",
    ],
    [
        ["nu e CC tot v7.1", "+", "nu mu CC tot v7.1", "+", "nu mu NC tot v7.1"],
        "nu total v7.1",
    ],  # my old PID selection
    [["muons7X8X v7.1", "+", "muons9X10X v7.1"],
        "muons v7.1"],  # my old PID selection
    [
        [
            "nu total v7.1",
            "+",
            "muons v7.1",
        ],
        "mc total v7.1",
    ],
]  # submenu to get all the quantities summed

for key, module in manager.results.items():
    if isinstance(module, ka.cutsHandler):
        # Print the cut handler name and the cut summary
        print("\n\t\t{}\n".format(module.name))
        module.printCuts()

    elif isinstance(module, ka.histogramHandler):
        # Create a TDirectory with the module name
        directory = outfile.mkdir(module.name)
        module.applyOperationMenu(
            sub_menu
        )  # <------------------ That's how you apply the sub_menu
        # export the histograms into directory
        module.exportToRoot(directory)

    elif isinstance(module, ka.exposureHandler):
        # Create a TDirectory with the module name
        directory = outfile.mkdir(module.name)
        module.exportToRoot(directory)


outfile.Close()

del manager

"""

manager_data.run()


import ROOT
#outfile = ROOT.TFile('mc_hists.root', 'RECREATE')
#outfile = ROOT.TFile('mc_hists_v7.2.root', 'RECREATE')
outfile = ROOT.TFile('data_hists.root', 'RECREATE')

sub_menu = [                                                                                                                                                                                                       
#        [['anu e CC low v7.2','+','anu e CC high v7.2','+','anu e CC higher v7.2','+','nu e CC low v7.2','+','nu e CC high v7.2','+','nu e CC higher v7.2'], 'nu e CC tot v7.2'], 
#        [['anu mu CC low v7.2', '+', 'anu mu CC high v7.2','+','anu mu CC higher v7.2','+', 'nu mu CC low v7.2', '+','nu mu CC high v7.2','+','nu mu CC higher v7.2'], 'nu mu CC tot v7.2'], 
#        [['anu mu NC low v7.2','+','anu mu NC high v7.2','+','anu mu NC higher v7.2','+','nu mu NC low v7.2','+','nu mu NC high v7.2','+','nu mu NC higher v7.2'], 'nu mu NC tot v7.2'], 
#        [['anu tau CC low v7.2','+','anu tau CC high v7.2','+','nu tau CC low v7.2','+','nu tau CC high v7.2'], 'nu tau CC tot v7.2'],
#        [['nu e CC tot v7.2','+','nu mu CC tot v7.2','+','nu mu NC tot v7.2', '+', 'nu tau CC tot v7.2'], 'nu tot v7.2'],  
#        [['nu total v7.2','+','muons v7.2',], 'mc tot v7.2'],   
#
#        [['anu e CC low v7.1','+','anu e CC high v7.1','+','anu e CC higher v7.1','+','nu e CC low v7.1','+','nu e CC high v7.1','+','nu e CC higher v7.1'], 'nu e CC tot v7.1'], 
#        [['anu mu CC higher v7.1','+','nu mu CC higher v7.1'], 'nu mu CC tot v7.1'], 
#        [['anu mu NC low v7.1','+','anu mu NC high v7.1','+','anu mu NC higher v7.1','+','nu mu NC low v7.1','+','nu mu NC high v7.1','+','nu mu NC higher v7.1'], 'nu mu NC tot v7.1'], 
#        [['nu e CC tot v7.1','+','nu mu CC tot v7.1','+','nu mu NC tot v7.1'], 'nu tot v7.1'],  # my old PID selection
#        [['muons7X8X v7.1','+','muons9X10X v7.1'], 'muons v7.1'],   # my old PID selection
#        [['nu total v7.1','+','muons v7.1',], 'mc tot v7.1'],   
#
#
#        [['nu e CC tot v7.1','+','nu e CC tot v7.2'], 'nu e CC tot'], 
#        [['nu mu CC tot v7.2'], 'nu mu CC tot'], 
#        [['nu mu NC tot v7.1','+','nu mu NC tot v7.2'], 'nu mu NC tot'], 
#        [['nu tau CC tot v7.2'], 'nu tau CC tot'], 
#        [['nu e CC tot','+','nu mu CC tot','+','nu mu NC tot', '+', 'nu tau CC tot'], 'nu tot'],  
#        [['muons v7.1','+','muons v7.2'], 'muons'], 
#        [['nu tot','+','muons',], 'mc tot'],   
        
 
        [['data 7X','+','data 80','+','data 85','+','data 90','+','data 95','+','data 100X','+','data 110X'], 'data', 'exposure_weighted'],


] # submenu to get all the quantities summed

for key, module in manager_data.results.items() :
    if isinstance(module, ka.cutsHandler):
        # Print the cut handler name and the cut summary
        print('\n\t\t{}\n'.format(module.name))
        module.printCuts()
        
    elif isinstance(module, ka.histogramHandler):
        # Create a TDirectory with the module name
        directory = outfile.mkdir(module.name)
        module.applyOperationMenu(sub_menu)#<------------------ That's how you apply the sub_menu
        # export the histograms into directory
        module.exportToRoot(directory)

    elif isinstance(module, ka.exposureHandler):
        # Create a TDirectory with the module name
        directory = outfile.mkdir(module.name)
        module.exportToRoot(directory)
    

outfile.Close()
"""
