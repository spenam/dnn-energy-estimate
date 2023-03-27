import pandas as pd # data analysis package
import numpy as np
import matplotlib.pyplot as plt
from functions import *
from plotting import *
import matplotlib
import h5py

def df_to_sarray(df):
    """
    Convert a pandas DataFrame object to a numpy structured array.
    This is functionally equivalent to but more efficient than
    np.array(df.to_array())

    :param df: the data frame to convert
    :return: a numpy structured array representation of df
    """

    v = df.values
    cols = df.columns
    types = [(cols[i].encode(), df[k].dtype.type) for (i, k) in enumerate(cols)]
    dtype = np.dtype(types)
    z = np.zeros(v.shape[0], dtype)
    for (i, k) in enumerate(z.dtype.names):
        z[k] = v[:, i]
    return z

plt.rcParams['figure.figsize'] = [10, 8]
font = {'size'   : 22}

matplotlib.rc('font', **font)


general_path = "/sps/km3net/users/spenamar/uproot_test/notebooks/data/"


#JGANDALF_path = general_path + "Neutrinos_JGANDALF_reco_merged_v7.10_for_ML_energy_estimation.h5"
JGANDALF_path = general_path + "Neutrinos_JGANDALF_JSHF_reco_merged_v7.10_for_Eest.h5"




# Load the hdf5 files

print("Loading JGANDALF ...")
#JGANDALF = pd.concat([x.query("run_id < 7400") for x in pd.read_hdf(JGANDALF_path)], ignore_index=True)
JGANDALF = pd.read_hdf(JGANDALF_path)
JGANDALFPRED = JGANDALF[JGANDALF['mc_energy_dst']<300]
JGANDALFPRED = JGANDALFPRED[JGANDALFPRED['mc_energy_dst']>0.3]
PRED = JGANDALFPRED[(JGANDALFPRED['run_id']<10000) & (JGANDALFPRED['run_id']>9000)]
#JGANDALF = JGANDALF[JGANDALF['mc_energy_dst']<100]
#JGANDALF = JGANDALF[JGANDALF['mc_energy_dst']>1]
JGANDALF = JGANDALF[JGANDALF['mc_energy_dst']<300]
JGANDALF = JGANDALF[JGANDALF['mc_energy_dst']>0.3]
JGANDALF = JGANDALF[(JGANDALF['run_id']>10000) | (JGANDALF['run_id']<9000)]
#JGANDALF = JGANDALF[JGANDALF['mc_pos_x']<500]
#JGANDALF = JGANDALF[JGANDALF['mc_pos_x']>400]
#JGANDALF = JGANDALF[JGANDALF['mc_pos_y']<600]
#JGANDALF = JGANDALF[JGANDALF['mc_pos_y']>550]
#JGANDALF = JGANDALF[JGANDALF['mc_pos_z']<200]
#JGANDALF = JGANDALF[JGANDALF['mc_pos_z']>0]
#JGANDALF = JGANDALF[JGANDALF['run_id']<9000]
#JGANDALF = JGANDALF[JGANDALF['run_id']<7400]
print(JGANDALF)
#JGANDALF = JGANDALF[JGANDALF['run_id']<8100]
#JGANDALF = JGANDALF[JGANDALF['run_id']>8000]
print("... done with JGANDALF")

#print(GNN.keys())

for i in [0]:
    #i = i+2


    if (i==0):
        Jcols = ['JSHOWERFIT_LENGTH_METRES','JSHOWERFIT_ENERGY','JENERGY_ENERGY','trig_hits','trig_doms','trig_lines','JSTART_LENGTH_METRES','mc_energy_dst',
         'mc_pos_x', 'mc_pos_y', 'mc_pos_z',
         'dir_x_gandalf', 'dir_y_gandalf', 'dir_z_gandalf', 'dir_x_showerfit', 'dir_y_showerfit',
                 'dir_z_showerfit','weights',
                'lik_first_JENERGY', 'lik_first_JSHOWERFIT',
                ] # Features to keep
        #Jcols = ['JSHOWERFIT_LENGTH_METRES','JSHOWERFIT_ENERGY','JENERGY_ENERGY','JSTART_LENGTH_METRES','mc_energy_dst',
        # 'dir_x_gandalf', 'dir_y_gandalf', 'dir_z_gandalf', 'dir_x_showerfit', 'dir_y_showerfit', 'dir_z_showerfit','weights'] # Features to keep
        Gcols = []
    #elif (i==1):
    #    Jcols = ['JSHOWERFIT_LENGTH_METRES','JSHOWERFIT_ENERGY','JENERGY_ENERGY','trig_hits','trig_doms','trig_lines','JSTART_LENGTH_METRES','mc_energy_dst',
    #     'mc_pos_x', 'mc_pos_y', 'mc_pos_z',
    #     'dir_x_gandalf', 'dir_y_gandalf', 'dir_z_gandalf', 'dir_x_showerfit', 'dir_y_showerfit', 'dir_z_showerfit','weights'] # Features to keep
    #    Gcols = ['pred_energy', 'pred_dir_x', 'pred_dir_y', 'pred_dir_z'] # Features to keep
    #elif (i==2):
    #    Jcols = ['JSHOWERFIT_LENGTH_METRES','JSHOWERFIT_ENERGY','JENERGY_ENERGY','trig_hits','trig_doms','trig_lines','JSTART_LENGTH_METRES','mc_energy_dst',
    #     'mc_pos_x', 'mc_pos_y', 'mc_pos_z',
    #     'dir_x_gandalf', 'dir_y_gandalf', 'dir_z_gandalf', 'dir_x_showerfit', 'dir_y_showerfit', 'dir_z_showerfit','weights', 'cherCond_n_doms', 'cherCond_n_doms_trig', 'cherCond_n_hits', 'cherCond_n_hits_trig',
    #             'cherCond_hits_meanZposition', 'cherCond_hits_trig_meanZposition', 'meanZhitTrig', 'gandalf_nHits',
    #             'gandalf_pos_r', 'showerfit_nHits', 'showerfit_pos_r'] # Features to keep
    #    Gcols = []
    #elif (i==3):
    #    Jcols = ['JSHOWERFIT_LENGTH_METRES','JSHOWERFIT_ENERGY','JENERGY_ENERGY','trig_hits','trig_doms','trig_lines','JSTART_LENGTH_METRES','mc_energy_dst',
    #     'mc_pos_x', 'mc_pos_y', 'mc_pos_z',
    #     'dir_x_gandalf', 'dir_y_gandalf', 'dir_z_gandalf', 'dir_x_showerfit', 'dir_y_showerfit', 'dir_z_showerfit','weights', 'cherCond_n_doms', 'cherCond_n_doms_trig', 'cherCond_n_hits', 'cherCond_n_hits_trig',
    #             'cherCond_hits_meanZposition', 'cherCond_hits_trig_meanZposition', 'meanZhitTrig', 'gandalf_nHits',
    #             'gandalf_pos_r', 'showerfit_nHits', 'showerfit_pos_r'] # Features to keep
    #    Gcols = ['pred_energy', 'pred_dir_x', 'pred_dir_y', 'pred_dir_z'] # Features to keep


    n_features = len(Jcols)+len(Gcols) - 5
    n_features = str(n_features)
    print( "THIS IS THE NUMBER OF FEATURES: "+str(n_features))
#    print( "THIS IS NN INFO: "+str(NN_info))

# Likelihood mask taking only tracks with possitive likelihood

    lik_mask = JGANDALF['lik_first_JENERGY']>0
    JGANDALF['lik_first_JENERGY'][lik_mask]
    lik_mask = lik_mask.to_list()


# Create dictionary with features from dst and GNN
# Make pandas dataframe

    small_dict = {}
    pred_dict = {}
    for j in Jcols:
        print(j)
        small_dict[j] = JGANDALF[j][lik_mask].tolist()
        pred_dict[j] = PRED[j].tolist()
#        print("Done with: " + j, file=open("outputs/"+NN_info+"/w/"+str(LR)+ "/output.txt","a"))
        
    for j in Gcols:
        print(j)
        small_dict[j] = GNN[j][lik_mask].tolist()
#        print("Done with: " + j, file=open("outputs/"+NN_info+"/w/"+str(LR)+ "/output.txt","a"))
            
    support_features = ['type', 'evt_id', 'run_id', 'frame_index', 'livetime_DAQ', 'ngen']
    for fe in support_features:
        print(fe)
        pred_dict[fe] = PRED[fe].tolist()
        small_dict[fe] = JGANDALF[fe][lik_mask].tolist()

    data = pd.DataFrame.from_dict(small_dict)
    pred = pd.DataFrame.from_dict(pred_dict)
    del small_dict
    del JGANDALF
    del JGANDALFPRED
    del pred_dict
    del PRED
    data = data.fillna(0)
    data['JENERGY_ENERGY'] = np.log10(data['JENERGY_ENERGY'])
    data['JSTART_LENGTH_METRES'] = np.log10(data['JSTART_LENGTH_METRES'])
    data['JSHOWERFIT_ENERGY'] = np.log10(data['JSHOWERFIT_ENERGY'])
    pred['JENERGY_ENERGY'] = np.log10(pred['JENERGY_ENERGY'])
    pred['JSTART_LENGTH_METRES'] = np.log10(pred['JSTART_LENGTH_METRES'])
    pred['JSHOWERFIT_ENERGY'] = np.log10(pred['JSHOWERFIT_ENERGY'])
    if (len(Gcols)!= 0):
        data['pred_energy'] = np.log10(data['pred_energy'])
        nan_mask = (data['JENERGY_ENERGY'] != -np.inf) & (data['JSHOWERFIT_ENERGY'] != -np.inf) & (data['pred_energy'] != -np.inf) & (data['JSTART_LENGTH_METRES'] != -np.inf)
    else:
        nan_mask = (data['JENERGY_ENERGY'] != -np.inf) & (data['JSHOWERFIT_ENERGY'] != -np.inf) & (data['JSTART_LENGTH_METRES'] != -np.inf)
    data = data[nan_mask]
    data = data.dropna(0)
    from sklearn.utils import shuffle
    data_shuffled = shuffle(data)#, random_state=3)
    #data_scaled = data_shuffled.drop(['weights','mc_energy_dst','mc_pos_x','mc_pos_y','mc_pos_z'],1)
    print(data_shuffled.keys())
    print(pred.keys())
    with h5py.File("shuffled_dsts.h5", "w") as f:
        f["data"] = data_shuffled.drop(support_features,1).to_records(index = False)
        f["support"] = data_shuffled[support_features].to_records(index = False)
    with h5py.File("for_pred.h5", "w") as f:
        f["data"] = pred.drop(support_features,1).to_records(index = False)
        f["support"] = pred[support_features].to_records(index = False)
    del pred
    del data_shuffled
    data = data.drop(support_features,1)
    data = data[data['mc_energy_dst']<100]
    data = data[data['mc_energy_dst']>1]
#    data = data[data['mc_energy_dst']<100]
#    data = data[data['mc_energy_dst']>1]
##    data = data[data['mc_pos_x']<500]
##    data = data[data['mc_pos_x']>400]
##    data = data[data['mc_pos_y']<600]
##    data = data[data['mc_pos_y']>550]
##    data = data[data['mc_pos_z']<200]
##    data = data[data['mc_pos_z']>0]
#    data = data[data['run_id']<10000]
#    data = data[data['run_id']>9000]

    X_all = data.drop(['mc_energy_dst','mc_pos_x','mc_pos_y','mc_pos_z'],1)
    y_all = np.log10(data['mc_energy_dst'])

# sampling of the histogram

    #uplim = 29000
    uplim = 300 # 600 #16000
    nbines = 50

    plt.clf()


    from sklearn.model_selection import train_test_split
    print(data)
    data_train, data_test = train_test_split(data, test_size = 0.3, random_state = 3)
    Ebins = np.logspace(0,2,nbines)
    plt.hist(data_train['mc_energy_dst'],bins = Ebins,weights = data_train['weights'])
    plt.xscale("log")
    plt.xlabel(r"$E_{true}$ [GeV]")
    plt.ylabel(r"Event counts")
    plt.tight_layout()
    plt.savefig("bins_pre_unbalanced.pdf")
    plt.savefig("bins_pre_unbalanced.png")
    #plt.show()
    plt.clf()
    print("pre-bootstrap")

    #cdata = bootstrap_sampling(data, uplim = uplim, lowE = 0, highE = 2, nbins = nbines)
    cdata = bootstrap_sampling(data_train, uplim = uplim, lowE = 0, highE = 1.3, nbins = int(nbines/2))
    cdatahigh = bootstrap_sampling(data_train, uplim = uplim/3, lowE = 1.3, highE = 2, nbins = int(nbines/2))
    cdata = cdata.append(cdatahigh, ignore_index=True)
    del cdatahigh
    print("after-bootstrap")

    plt.hist(cdata['mc_energy_dst'],bins = Ebins,weights = cdata['weights'])
    plt.xscale("log")
    plt.xlabel(r"$E_{true}$ [GeV]")
    plt.ylabel(r"Event counts")
    plt.tight_layout()
    plt.savefig("bins_after_unbalanced.pdf")
    plt.savefig("bins_after_unbalanced.png")
    #plt.show()
    plt.clf()
    del data


    X_all = cdata.drop(['mc_energy_dst','mc_pos_x','mc_pos_y','mc_pos_z'],1)
    if (len(Gcols)!= 0):
        #nan_mask = (X_all['JENERGY_ENERGY'] != -np.inf) & (X_all['pred_energy'] != -np.inf) &        (X_all['JSTART_LENGTH_METRES'] != -np.inf)
        nan_mask = (X_all['JENERGY_ENERGY'] != -np.inf) & (X_all['JSHOWERFIT_ENERGY'] != -np.inf) & (X_all['pred_energy'] != -np.inf) &        (X_all['JSTART_LENGTH_METRES'] != -np.inf)
    else: #nan_mask = (X_all['JENERGY_ENERGY'] != -np.inf) & (X_all['JSTART_LENGTH_METRES'] != -np.inf)
        nan_mask = (X_all['JENERGY_ENERGY'] != -np.inf) & (X_all['JSHOWERFIT_ENERGY'] != -np.inf) & (X_all['JSTART_LENGTH_METRES'] != -np.inf)
    X_all = X_all[nan_mask]
    y_all = np.log10(cdata['mc_energy_dst'])
    y_all = y_all[nan_mask]

    del cdata

# splitting data

    #from sklearn.model_selection import train_test_split
    #X_train, X_test, y_train, y_test = train_test_split(X_all, y_all,test_size = 0.3, random_state = 3)
    X_train = X_all
    y_train = y_all
    X_test = data_test.drop(['mc_energy_dst','mc_pos_x','mc_pos_y','mc_pos_z'],1)
    y_test = np.log10(data_test['mc_energy_dst'])

    X_train_scaled = X_train.drop(['weights'],1)
    X_test_scaled = X_test.drop(['weights'],1)

    X_train_scaled = X_train_scaled.to_records(index=False)

    X_test_scaled = X_test_scaled.to_records(index=False)

    #X_train_scaled.to_hdf('for_train_nonbalanced.h5', key='X')
    #y_train.to_hdf('for_train_nonbalanced.h5', key='y', index = False)
    #X_train['weights'].to_hdf('for_train_nonbalanced.h5', key='weights', index = False)

    #X_test_scaled.to_hdf('for_val_nonbalanced.h5', key='X')
    #y_test.to_hdf('for_val_nonbalanced.h5', key='y', index = False)
    #X_test['weights'].to_hdf('for_val_nonbalanced.h5', key='weights', index = False)
    with h5py.File("for_train.h5", "w") as f:
        f['X'] = X_train_scaled
        f['y'] = y_train
        f['weights'] = X_train['weights']
    with h5py.File("for_val.h5", "w") as f:
        f['X'] = X_test_scaled
        f['y'] = y_test
        f['weights'] = X_test['weights']

