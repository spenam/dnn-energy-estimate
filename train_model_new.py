import pandas as pd  # data analysis package
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import pickle
from functions import *
from plotting import *
from time import time
from sklearn.neural_network import MLPRegressor
import matplotlib
import matplotlib as mpl
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras import optimizers
import h5py


def plot_loss(history):
    plt.plot(history.history["loss"], label="loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    # plt.ylim([0,10])
    plt.xlabel("Epoch")
    plt.ylabel("Error (Loss)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/" + NN_info + "/w/" + str(LR) + "/loss_" + n_features + ".pdf")
    plt.savefig("plots/" + NN_info + "/w/" + str(LR) + "/loss_" + n_features + ".png")
    plt.grid(True)
    # plt.show()
    plt.clf()


plt.rcParams["figure.figsize"] = [10, 8]
font = {"size": 22}

matplotlib.rc("font", **font)
general_path = "/sps/km3net/users/spenamar/uproot_test/notebooks/data/"


# JGANDALF_path = general_path + "Neutrinos_JGANDALF_reco_merged_v7.10_for_ML_energy_estimation.h5"
JGANDALF_path = general_path + "Neutrinos_JGANDALF_JSHF_reco_merged_v7.10_for_Eest.h5"


parser = argparse.ArgumentParser(description="NN architecture")
#parser.add_argument(
#    "-nn", "--NN", nargs="+", help="Architecture of the neural network", required=True
#)
parser.add_argument(
    "-nl", "--NL", type = int, help="Number of layers in the network", required=True
)
parser.add_argument(
    "-ls", "--LS", type = int, help="Size of each layer", required=True
)
parser.add_argument(
    "-lkl", "--likelihood", type = int, help="0 for training without likelihood and 5 for training with", required=True
)
parser.add_argument(
    "-lf", "--LossFunction", type = str, help="Loss function to use", required=True
)
#parser.add_argument(
#    "-b", "--batchnorm", type=int, help="batchnorm boolean value", required=True
#)
# parser.add_argument('-d','--dropout', type=int,  help='dropout boolean value', required=True)
parser.add_argument(
    "-lr",
    "--learning_rate",
    type=float,
    help="learning rate float value",
    required=True,
)
args = parser.parse_args()
arch = [int(x) for x in [args.LS]*args.NL]
#batchnorm = args.batchnorm
# drop = args.dropout
batchnorm = False #myparam[0]
drop = 0  # myparam[1]
LR = args.learning_rate
lkl = args.likelihood
lossf = args.LossFunction
drop_vals = 0.3 * np.ones(len(arch))



# print(GNN.keys())

for i in [lkl]:
    # i = i+2

    # Establish the model

    # learning_rate=0.000001
    model = Sequential()
    batchnorm_str = ""
    dropout_str = ""
    model.add(layers.BatchNormalization())

    for j in range(len(arch)):
        model.add(layers.Dense(arch[j]))
        if batchnorm == True:
            batchnorm_str = "_bn_"
            model.add(layers.BatchNormalization())
        model.add(layers.Activation("PReLU"))
        if drop == True:
            dropout_str = "_do_"
            model.add(layers.Dropout(drop_vals[j]))
    model.add(layers.Dense(1))

    # LR = 0.000001

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=LR), loss=lossf)

    # The NN info
    NN_info = "_".join(str(x) for x in arch)
    print(NN_info)
    NN_info = NN_info + "_" + lossf + "_v7.10_jshf_"
    if not os.path.exists("plots"):
        os.makedirs("plots")
    if not os.path.exists("plots/" + NN_info):
        os.makedirs("plots/" + NN_info)
    if not os.path.exists("plots/" + NN_info + "/w"):
        os.makedirs("plots/" + NN_info + "/w")
    if not os.path.exists("plots/" + NN_info + "/w/" + str(LR)):
        os.makedirs("plots/" + NN_info + "/w/" + str(LR))
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    if not os.path.exists("outputs/" + NN_info):
        os.makedirs("outputs/" + NN_info)
    if not os.path.exists("outputs/" + NN_info + "/w"):
        os.makedirs("outputs/" + NN_info + "/w")
    if not os.path.exists("outputs/" + NN_info + "/w/" + str(LR)):
        os.makedirs("outputs/" + NN_info + "/w/" + str(LR))

    if i == 0:
        Jcols = [
            "JSHOWERFIT_LENGTH_METRES",
            "JSHOWERFIT_ENERGY",
            "JENERGY_ENERGY",
            "trig_hits",
            "trig_doms",
            "trig_lines",
            "JSTART_LENGTH_METRES",
            "mc_energy_dst",
            "mc_pos_x",
            "mc_pos_y",
            "mc_pos_z",
            "dir_x_gandalf",
            "dir_y_gandalf",
            "dir_z_gandalf",
            "dir_x_showerfit",
            "dir_y_showerfit",
            "dir_z_showerfit",
            "weights",
        ]  # Features to keep
        # Jcols = ['JSHOWERFIT_LENGTH_METRES','JSHOWERFIT_ENERGY','JENERGY_ENERGY','JSTART_LENGTH_METRES','mc_energy_dst',
        # 'dir_x_gandalf', 'dir_y_gandalf', 'dir_z_gandalf', 'dir_x_showerfit', 'dir_y_showerfit', 'dir_z_showerfit','weights'] # Features to keep
        Gcols = []
    elif i == 5:
        Jcols = [
            "JSHOWERFIT_LENGTH_METRES",
            "JSHOWERFIT_ENERGY",
            "JENERGY_ENERGY",
            "lik_first_JENERGY",
            "lik_first_JSHOWERFIT",
            "trig_hits",
            "trig_doms",
            "trig_lines",
            "JSTART_LENGTH_METRES",
            "mc_energy_dst",
            "mc_pos_x",
            "mc_pos_y",
            "mc_pos_z",
            "dir_x_gandalf",
            "dir_y_gandalf",
            "dir_z_gandalf",
            "dir_x_showerfit",
            "dir_y_showerfit",
            "dir_z_showerfit",
            "weights",
        ]  # Features to keep
        # Jcols = ['JSHOWERFIT_LENGTH_METRES','JSHOWERFIT_ENERGY','JENERGY_ENERGY','JSTART_LENGTH_METRES','mc_energy_dst',
        # 'dir_x_gandalf', 'dir_y_gandalf', 'dir_z_gandalf', 'dir_x_showerfit', 'dir_y_showerfit', 'dir_z_showerfit','weights'] # Features to keep
        Gcols = []
    elif i == 1:
        Jcols = [
            "JSHOWERFIT_LENGTH_METRES",
            "JSHOWERFIT_ENERGY",
            "JENERGY_ENERGY",
            "trig_hits",
            "trig_doms",
            "trig_lines",
            "JSTART_LENGTH_METRES",
            "mc_energy_dst",
            "mc_pos_x",
            "mc_pos_y",
            "mc_pos_z",
            "dir_x_gandalf",
            "dir_y_gandalf",
            "dir_z_gandalf",
            "dir_x_showerfit",
            "dir_y_showerfit",
            "dir_z_showerfit",
            "weights",
        ]  # Features to keep
        Gcols = [
            "pred_energy",
            "pred_dir_x",
            "pred_dir_y",
            "pred_dir_z",
        ]  # Features to keep
    elif i == 2:
        Jcols = [
            "JSHOWERFIT_LENGTH_METRES",
            "JSHOWERFIT_ENERGY",
            "JENERGY_ENERGY",
            "trig_hits",
            "trig_doms",
            "trig_lines",
            "JSTART_LENGTH_METRES",
            "mc_energy_dst",
            "mc_pos_x",
            "mc_pos_y",
            "mc_pos_z",
            "dir_x_gandalf",
            "dir_y_gandalf",
            "dir_z_gandalf",
            "dir_x_showerfit",
            "dir_y_showerfit",
            "dir_z_showerfit",
            "weights",
            "cherCond_n_doms",
            "cherCond_n_doms_trig",
            "cherCond_n_hits",
            "cherCond_n_hits_trig",
            "cherCond_hits_meanZposition",
            "cherCond_hits_trig_meanZposition",
            "meanZhitTrig",
            "gandalf_nHits",
            "gandalf_pos_r",
            "showerfit_nHits",
            "showerfit_pos_r",
        ]  # Features to keep
        Gcols = []
    elif i == 3:
        Jcols = [
            "JSHOWERFIT_LENGTH_METRES",
            "JSHOWERFIT_ENERGY",
            "JENERGY_ENERGY",
            "trig_hits",
            "trig_doms",
            "trig_lines",
            "JSTART_LENGTH_METRES",
            "mc_energy_dst",
            "mc_pos_x",
            "mc_pos_y",
            "mc_pos_z",
            "dir_x_gandalf",
            "dir_y_gandalf",
            "dir_z_gandalf",
            "dir_x_showerfit",
            "dir_y_showerfit",
            "dir_z_showerfit",
            "weights",
            "cherCond_n_doms",
            "cherCond_n_doms_trig",
            "cherCond_n_hits",
            "cherCond_n_hits_trig",
            "cherCond_hits_meanZposition",
            "cherCond_hits_trig_meanZposition",
            "meanZhitTrig",
            "gandalf_nHits",
            "gandalf_pos_r",
            "showerfit_nHits",
            "showerfit_pos_r",
        ]  # Features to keep
        Gcols = [
            "pred_energy",
            "pred_dir_x",
            "pred_dir_y",
            "pred_dir_z",
        ]  # Features to keep

    n_features = len(Jcols) + len(Gcols) - 5
    n_features = str(n_features)
    print(
        "THIS IS THE NUMBER OF FEATURES: " + str(n_features),
        file=open("outputs/" + NN_info + "/w/" + str(LR) + "/output.txt", "a"),
    )
    print("THIS IS THE NUMBER OF FEATURES: " + str(n_features))
    print("THIS IS NN INFO: " + str(NN_info))

    with h5py.File("for_train.h5", "r") as f:
        with h5py.File("for_val.h5", "r") as g:
            print("Opening hdf5...")

            if lkl== 0:
                X_train_scaled = tf.convert_to_tensor(pd.DataFrame(f["X"]['JSHOWERFIT_LENGTH_METRES', 'JSHOWERFIT_ENERGY', 'JENERGY_ENERGY', 'trig_hits', 'trig_doms', 'trig_lines', 'JSTART_LENGTH_METRES', 'dir_x_gandalf', 'dir_y_gandalf', 'dir_z_gandalf','dir_x_showerfit', 'dir_y_showerfit', 'dir_z_showerfit']).to_numpy())
                X_val_scaled = tf.convert_to_tensor(pd.DataFrame(g["X"]['JSHOWERFIT_LENGTH_METRES', 'JSHOWERFIT_ENERGY', 'JENERGY_ENERGY', 'trig_hits', 'trig_doms', 'trig_lines', 'JSTART_LENGTH_METRES', 'dir_x_gandalf', 'dir_y_gandalf', 'dir_z_gandalf','dir_x_showerfit', 'dir_y_showerfit', 'dir_z_showerfit']).to_numpy())
            elif lkl == 5:
                X_train_scaled = tf.convert_to_tensor(pd.DataFrame(f["X"]['JSHOWERFIT_LENGTH_METRES', 'JSHOWERFIT_ENERGY',
                                                                          'JENERGY_ENERGY', 'lik_first_JENERGY',
                                                                          'lik_first_JSHOWERFIT', 'trig_hits', 'trig_doms', 'trig_lines', 'JSTART_LENGTH_METRES', 'dir_x_gandalf', 'dir_y_gandalf', 'dir_z_gandalf','dir_x_showerfit', 'dir_y_showerfit', 'dir_z_showerfit']).to_numpy())
                X_val_scaled = tf.convert_to_tensor(pd.DataFrame(g["X"]['JSHOWERFIT_LENGTH_METRES', 'JSHOWERFIT_ENERGY',
                                                                          'JENERGY_ENERGY', 'lik_first_JENERGY',
                                                                          'lik_first_JSHOWERFIT', 'trig_hits', 'trig_doms', 'trig_lines', 'JSTART_LENGTH_METRES', 'dir_x_gandalf', 'dir_y_gandalf', 'dir_z_gandalf','dir_x_showerfit', 'dir_y_showerfit', 'dir_z_showerfit']).to_numpy())
            #X_train_scaled = tf.convert_to_tensor(pd.DataFrame(f["X"][:]).to_numpy())
            y_train = np.asarray(f["y"])
            X_train_weights = np.asarray(f["weights"])
            #X_val_scaled = tf.convert_to_tensor(pd.DataFrame(g["X"][:]).to_numpy())
            y_val = np.asarray(g["y"])
            X_val_weights = np.asarray(g["weights"])
            print("finished opening hdf5...")

            start = time()
            #history = model.fit_generator(
            history = model.fit(
                #(X_train_scaled, y_train, X_train_weights), # input to fit_generator
                epochs=20,
                x=X_train_scaled,
                y=y_train,
                # history = model.fit(epochs=1, x=X_train_scaled, y=y_train,
                validation_data=(X_val_scaled, y_val, X_val_weights),
                verbose=2,
                sample_weight=X_train_weights,
                #shuffle=False,
            )
            end = time()
            print(
                "Training time in {:4f} seconds".format(end - start),
                file=open("outputs/" + NN_info + "/w/" + str(LR) + "/output.txt", "a"),
            )
            plot_loss(history)

        if not os.path.exists("models_w"):
            os.makedirs("models_w")
        model.save("models_w/" + NN_info + "_w_" + n_features + "_" + str(LR) + ".h5")

