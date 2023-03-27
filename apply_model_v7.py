import numpy as np
import pandas as pd
import uproot
import km3pipe as kp
import km3ant as ka
import argparse


# Small trick to avoid having the terminal flooded by
# non-sense tensorflow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

parser = argparse.ArgumentParser(description='File name')

parser.add_argument("-fn",
                                      '--filename',
                                      help="file name.",
                                      type=str,
                                      required=True)
arguments = parser.parse_args()

fname = str(arguments.filename)



dataFileDir = '/sps/km3net/users/alflazo/dstProd/v7.1_v7.2_jsh/' # dst-alfonso




outFilePath = './results/v7.10/'
if not os.path.exists(outFilePath):
    os.makedirs(outFilePath, exist_ok=True)
outFileExtension = 'DNN'
outFileName = outFilePath+'hist_'+outFileExtension+'.root' # Output file for the histograms
# Declare datasets
dataSets = {
    fname.split('.root')[0] : {'filename':dataFileDir+fname}
}
print(dataFileDir + fname)

for key, items in dataSets.items():                                                                                                                                                                                
    items["pipename"] = key       # Don't remember why but that's important 
    items['friend_trees'] = ['T']  # Tells the pump to load the tree "T" and "<path_to_the_file>:gnn", in addition to the E tree


# Weighting class from km3ant

# set the weighing scheme :
for key, value in dataSets.items():
    
    #value['ext_weight'] = ''
    #value['func_weight'] = None
    #value['func_weight_nosc'] = None
    #value['func_kwargs'] = None
    
    #if key.find('data') != -1 :  # Data, will use cst weight of 1
    #    pass
    #    
    #else: # Atm. muons, use pre-computed weight from DST (livetime_sim/livetime_DAQ, per run)
    #    value['ext_weight'] = 'T.sum_mc_evt.weight'
        
    # Set tree output file, storing cuts results etc ...
    print(outFilePath)
    print(key.replace(' ', '-'))
    value['output_filename'] = '{}/{}_{}.root'.format(outFilePath,
                                                                 key.replace(' ','-'),
                                                                 outFileExtension)







# Open the output root file


# Declare the shape of the pipe 
manager = ka.pipeManager(dataSets, step_size = 1000000, verbosity='WARNING', timeit = True)#, kwargs_pump = {'export_header_target': outFile.mkdir('headerTrees')})
manager.append_module("ant_pump", ka.AntPump, kwargs = {"step_size":1000000}, dataset_fields = {"filename":"filename", "friend_trees":"friend_trees", "pipename":"pipename"})

#manager.append_module("exposure", ka.exposureHandler)

from model_keras_v7 import apply_model

manager.append_module("apply_model", apply_model, kwargs =
                      # std
                      {"input_dir":"/sps/km3net/users/spenamar/hackathon/tf/training_workflow/models_w/32_32_32_32_32_32_32_32_32_32_32_32_log_cosh_v7.10_jshf__w_15_1e-05.h5","output_key":"energy_recoDNN",
                       "pwd":"."})


# Compute exposure time for the given run list
# Write the output file
manager.append_module("output_writer", ka.outputTreeHandler,
                      dataset_fields = {'filename':'output_filename'},
                      kwargs = {'masks':[],
                                'branches':["energy_recoDNN"],
                                'treename': 'DNN'})


# Chewbie, prepare the light speed engine 
manager.run() # FUSHHHHHH *Unrealistic faster-than-light speed sound*

#manager.export_timeit('timeit_info.p')
exit()
