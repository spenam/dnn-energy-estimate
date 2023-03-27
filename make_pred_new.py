import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np
from time import time
import h5py
import argparse
import os

parser = argparse.ArgumentParser(description="NN architecture")
parser.add_argument(
    "-nl", "--NL", type=int, help="Number of layers in the network", required=True
)
parser.add_argument("-ls", "--LS", type=int,
                    help="Size of each layer", required=True)
parser.add_argument(
    "-nf", "--NF", type=int, help="Number of training features", required=True
)
parser.add_argument(
    "-lkl",
    "--likelihood",
    type=int,
    help="0 for training without likelihood and 5 for training with",
    required=True,
)
parser.add_argument(
    "-lf", "--LossFunction", type=str, help="Loss function to use", required=True
)
# parser.add_argument(
#    "-b", "--batchnorm", type=int, help="batchnorm boolean value", required=True
# )
# parser.add_argument('-d','--dropout', type=int,  help='dropout boolean value', required=True)
parser.add_argument(
    "-lr",
    "--learning_rate",
    type=float,
    help="learning rate float value",
    required=True,
)
args = parser.parse_args()


args = parser.parse_args()
arch = [int(x) for x in [args.LS] * args.NL]
# batchnorm = args.batchnorm
# drop = args.dropout
batchnorm = False  # myparam[0]
drop = 0  # myparam[1]
LR = args.learning_rate
lkl = args.likelihood
n_features = args.NF
lossf = args.LossFunction
drop_vals = 0.3 * np.ones(len(arch))

# Load model
NN_info = "_".join(str(x) for x in arch)
print(NN_info)
NN_info = NN_info + "_" + lossf + "_v7.10_jshf_"

model = keras.models.load_model(
    "models_w/" + NN_info + "_w_" + str(n_features) + "_" + str(LR) + ".h5"
)


# Prediction

# with h5py.File("v7.2/shuffled_dsts.h5", "r") as f:
with h5py.File("shuffled_dsts.h5", "r") as f:
    full_data = f["data"]
    if lkl == 0:
        data_scaled = tf.convert_to_tensor(
            pd.DataFrame(
                full_data[
                    "JSHOWERFIT_LENGTH_METRES",
                    "JSHOWERFIT_ENERGY",
                    "JENERGY_ENERGY",
                    "trig_hits",
                    "trig_doms",
                    "trig_lines",
                    "JSTART_LENGTH_METRES",
                    "dir_x_gandalf",
                    "dir_y_gandalf",
                    "dir_z_gandalf",
                    "dir_x_showerfit",
                    "dir_y_showerfit",
                    "dir_z_showerfit",
                ]
            ).to_numpy()
        )
    elif lkl == 5:
        data_scaled = tf.convert_to_tensor(
            pd.DataFrame(
                full_data[
                    "JSHOWERFIT_LENGTH_METRES",
                    "JSHOWERFIT_ENERGY",
                    "JENERGY_ENERGY",
                    "lik_first_JENERGY",
                    "lik_first_JSHOWERFIT",
                    "trig_hits",
                    "trig_doms",
                    "trig_lines",
                    "JSTART_LENGTH_METRES",
                    "dir_x_gandalf",
                    "dir_y_gandalf",
                    "dir_z_gandalf",
                    "dir_x_showerfit",
                    "dir_y_showerfit",
                    "dir_z_showerfit",
                ]
            ).to_numpy()
        )
    start = time()
    y_pred = model.predict(data_scaled)
    end = time()
    print(
        "prediction time in {:4f} seconds".format(end - start),
        # file=open("outputs/" + NN_info + "/w/" + str(LR) + "/output.txt", "a"),
    )

    if not os.path.exists("pred_y"):
        os.makedirs("pred_y")
    with h5py.File(
        # "pred_y/" + NN_info + "_w_allDSTs_" + str(n_features) + "_" + str(LR) + "_v7.2DSTs.h5", "w"
        "pred_y/" + NN_info + "_w_allDSTs_" + \
            str(n_features) + "_" + str(LR) + ".h5",
        "w",
    ) as g:
        # g.create_dataset("y", data=y_pred)
        g["y"] = np.log10(f["data"]["mc_energy_dst"])
        g["y_pred"] = y_pred
        g["JENERGY"] = f["data"]["JENERGY_ENERGY"]
        g["JSHOWERFIT"] = f["data"]["JSHOWERFIT_ENERGY"]
        g["JSTART"] = f["data"]["JSTART_LENGTH_METRES"]
        g["JSHOWERFIT_LENGTH"] = f["data"]["JSHOWERFIT_LENGTH_METRES"]
        g["weights"] = f["data"]["weights"]
        g["support"] = np.copy(f["support"])

with h5py.File("for_pred.h5", "r") as f:
    full_data = f["data"]
    if lkl == 0:
        data_scaled = tf.convert_to_tensor(
            pd.DataFrame(
                full_data[
                    "JSHOWERFIT_LENGTH_METRES",
                    "JSHOWERFIT_ENERGY",
                    "JENERGY_ENERGY",
                    "trig_hits",
                    "trig_doms",
                    "trig_lines",
                    "JSTART_LENGTH_METRES",
                    "dir_x_gandalf",
                    "dir_y_gandalf",
                    "dir_z_gandalf",
                    "dir_x_showerfit",
                    "dir_y_showerfit",
                    "dir_z_showerfit",
                ]
            ).to_numpy()
        )
    elif lkl == 5:
        data_scaled = tf.convert_to_tensor(
            pd.DataFrame(
                full_data[
                    "JSHOWERFIT_LENGTH_METRES",
                    "JSHOWERFIT_ENERGY",
                    "JENERGY_ENERGY",
                    "lik_first_JENERGY",
                    "lik_first_JSHOWERFIT",
                    "trig_hits",
                    "trig_doms",
                    "trig_lines",
                    "JSTART_LENGTH_METRES",
                    "dir_x_gandalf",
                    "dir_y_gandalf",
                    "dir_z_gandalf",
                    "dir_x_showerfit",
                    "dir_y_showerfit",
                    "dir_z_showerfit",
                ]
            ).to_numpy()
        )
    start = time()
    y_pred = model.predict(data_scaled)
    end = time()
    print(
        "prediction time in {:4f} seconds".format(end - start),
        # file=open("outputs/" + NN_info + "/w/" + str(LR) + "/output.txt", "a"),
    )

    if not os.path.exists("pred_y"):
        os.makedirs("pred_y")
    with h5py.File(
        # "pred_y/" + NN_info + "_w_blindDSTs_" + str(n_features) + "_" + str(LR) + "_v7.2DSTs.h5", "w"
        "pred_y/" + NN_info + "_w_blindDSTs_" + \
            str(n_features) + "_" + str(LR) + ".h5",
        "w",
    ) as g:
        # g.create_dataset("y", data=y_pred)
        g["y"] = np.log10(f["data"]["mc_energy_dst"])
        g["JENERGY"] = f["data"]["JENERGY_ENERGY"]
        g["JSHOWERFIT"] = f["data"]["JSHOWERFIT_ENERGY"]
        g["JSTART"] = f["data"]["JSTART_LENGTH_METRES"]
        g["JSHOWERFIT_LENGTH"] = f["data"]["JSHOWERFIT_LENGTH_METRES"]
        g["y_pred"] = y_pred
        g["weights"] = f["data"]["weights"]
        g["support"] = np.copy(f["support"])
