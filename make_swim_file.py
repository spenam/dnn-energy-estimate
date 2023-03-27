import km3ant as ka
import argparse
from swim_def import (
    Add_SWIMinfo_neutrino,
    Add_SWIMinfo_muon,
    Add_SWIMinfo_JSHF,
    Add_SWIMinfo_DNN,
)


# Small trick to avoid having the terminal flooded by
# non-sense tensorflow warnings
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

parser = argparse.ArgumentParser(description="File name")

parser.add_argument("-fn", "--filename", help="file name.",
                    type=str, required=True)
arguments = parser.parse_args()

fname = str(arguments.filename)


dataFileDir = "/sps/km3net/users/alflazo/dstProd/v7.1_v7.2_jsh/"  # dst-alfonso
# friend_files_directory = "/sps/km3net/users/alflazo/dstProd/SWIM_file_tmp/v7.1_7.2_jsh"
friend_files_directory = "/sps/km3net/users/spenamar/oscillations/PIDfilesAlfonso"
friend_files_directory2 = "/sps/km3net/users/spenamar/hackathon/tf/training_workflow/model_pipeline/results/v7.10"

# Swim branches names

branches_swim = [
    "type",
    "run_id",
    "run_duration",
    "mc_id",
    "is_cc",
    "is_neutrino",
    "pos_x_true",
    "pos_y_true",
    "pos_z_true",
    "pos_x_recoJGandalf",
    "pos_y_recoJGandalf",
    "pos_z_recoJGandalf",
    "pos_x_recoJShower",
    "pos_y_recoJShower",
    "pos_z_recoJShower",
    "pos_x_recoDusj",
    "pos_y_recoDusj",
    "pos_z_recoDusj",
    "energy_true",
    "cos_zenith_true",
    "bjorken_y_true",
    "energy_recoJEnergy",
    "energy_recoTracklength",
    "cos_zenith_recoJGandalf",
    "bjorken_y_recoJGandalf",
    "energy_recoDusj",
    "cos_zenith_recoDusj",
    "bjorken_y_recoDusj",
    "energy_recoJShower",
    "energy_recoDNN",
    "energy_recoRatioEL_JEnergy",
    "energy_recoRatioLE_JEnergy",
    "energy_recoRatioEL_Tracklen",
    "energy_recoRatioLE_Tracklen",
    "energy_recoRatioEL_JShower",
    "energy_recoRatioLE_JShower",
    "energy_recoRatioEL_DNN",
    "energy_recoRatioLE_DNN",
    "cos_zenith_recoJShower",
    "bjorken_y_recoJShower",
    "gandalf_lik",
    "JShower_lik",
    "w2",
    "w1",
    "ngen",
    "int_len",
    "EG",
    "E_min_gen",
    "E_max_gen",
    "weight_one_year",
    "pid_proba_track",
    "antimu_proba_bkg",
    "nTriggerHits",
    "gandalf_pos_r",
    "maximumToT_triggerHit",
    "beta0",
    "meanZhitTrig",
    "rectype_JShower",
    "rectype_JGandalf",
]

# ICRC_cuts       = 1
# ICRC_V2_cuts    = 0
# ICRC_V2_100GeV_cuts = 0
# ICRC_V2_50GeV_cuts = 1
# ICRC_V2_20GeV_cuts = 1


outFilePath = "./results/v7.10/"
if not os.path.exists(outFilePath):
    os.makedirs(outFilePath)
outFileExtension = "DNN"
# Declare datasets
dataSets = {fname.split(".root")[0]: {"filename": dataFileDir + fname}}
print(dataFileDir + fname)

for key, items in dataSets.items():
    items["pipename"] = key  # Don't remember why but that's important
    fname = items["filename"].split("/")[
        -1
    ]  # Get the DST filename to convert it into GNN file name
    # fname = 'SelectedEvents_PID_tree_' + fname + '.root'
    fname = "outputTree_PID_" + fname + ".root"
    items["filename_friend"] = friend_files_directory + \
        "/" + fname.split("/")[-1]
    # items['friend_trees'] = ['T',items['filename_friend']+':events']
    fname = (
        items["filename"].split("/")[-1].split(".root")[0]
    )  # Get the DST filename to convert it into GNN file name
    fname = fname + "_DNN.root"
    items["filename_friend2"] = friend_files_directory2 + \
        "/" + fname.split("/")[-1]
    # items['friend_trees'] = ['T',items['filename_friend']+':events']
    # items['friend_trees'] = ['T',items['filename_friend']+':tree']
    # items['friend_trees'] = ['T',items['filename_friend']+':sel',items['filename_friend2']+':tree']  # Tells the pump to load the tree "T" and "<path_to_the_file>:gnn", in addition to the E tree name":<dstfilepath>}``
    # items['friend_trees'] = ['T',items['filename_friend2']+':tree']  # Tells the pump to load the tree "T" and "<path_to_the_file>:gnn", in addition to the E tree name":<dstfilepath>}``
    items["friend_trees"] = [
        "T",
        items["filename_friend"] + ":tree",
        items["filename_friend2"] + ":DNN",
    ]  # Tells the pump to load the tree "T" and "<path_to_the_file>:gnn", in addition to the E tree name":<dstfilepath>}``
    print(items["friend_trees"])


# Weighting class from km3ant

# set the weighing scheme :
for key, value in dataSets.items():
    print(outFilePath)
    print(key.replace(" ", "-"))
    value["output_filename"] = "{}/SelectedEvents_{}_{}.root".format(
        outFilePath, key.replace(" ", "-"), outFileExtension
    )


# Open the output root file


# Declare the shape of the pipe
manager = ka.pipeManager(
    dataSets, step_size=1000000, verbosity="WARNING", timeit=True
)  # , kwargs_pump = {'export_header_target': outFile.mkdir('headerTrees')})
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
precuts = [
    ka.lin_cut("E.trks.dir.z[:,0]", ">", 0),
    ka.lin_cut("E.trks.lik[:,0]", ">", 40),
    ka.lin_cut("T.sum_trig_hits.nhits", ">", 15),
    # ka.lin_cut('used_train','==',0),
]


manager.append_module("Add_SWIMinfo_JSHF", Add_SWIMinfo_JSHF)
manager.append_module("Add_SWIMinfo_DNN", Add_SWIMinfo_DNN)
if "mupage" not in str(fname):
    manager.append_module("Add_SWIMinfo_neutrino", Add_SWIMinfo_neutrino)
else:
    manager.append_module("Add_SWIMinfo_muon", Add_SWIMinfo_muon)
manager.append_module(
    "pre_cuts", ka.cutsHandler, kwargs={"cuts": precuts, "apply_mask": True}
)


# Compute exposure time for the given run list
# Write the output file
manager.append_module(
    "output_writer",
    ka.outputTreeHandler,
    dataset_fields={"filename": "output_filename"},
    kwargs={"masks": [], "branches": branches_swim, "treename": "sel"},
)

# Chewbie, prepare the light speed engine
manager.run()  # FUSHHHHHH *Unrealistic faster-than-light speed sound*

# manager.export_timeit('timeit_info.p')
exit()
