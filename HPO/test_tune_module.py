import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import tune_module
import numpy as np
import pandas as pd
import simpler_tune_module as stm
import keras_tuner as kt


def get_dataset():#fpath: str):
    with h5py.File("../../for_train.h5", "r") as f:
        with h5py.File("../../for_val.h5", "r") as g:
            print("Opening hdf5...")
            X_train_scaled = tf.convert_to_tensor(pd.DataFrame(f["X"]['JSHOWERFIT_ENERGY',
                                                                          'JENERGY_ENERGY', 'lik_first_JENERGY',
                                                                          'lik_first_JSHOWERFIT', 'trig_hits', 'trig_doms', 'trig_lines', 'JSTART_LENGTH_METRES', 'dir_x_gandalf', 'dir_y_gandalf', 'dir_z_gandalf','dir_x_showerfit', 'dir_y_showerfit', 'dir_z_showerfit']).to_numpy())
            X_val_scaled = tf.convert_to_tensor(pd.DataFrame(g["X"]['JSHOWERFIT_ENERGY',
                                                                          'JENERGY_ENERGY', 'lik_first_JENERGY',
                                                                          'lik_first_JSHOWERFIT', 'trig_hits', 'trig_doms', 'trig_lines', 'JSTART_LENGTH_METRES', 'dir_x_gandalf', 'dir_y_gandalf', 'dir_z_gandalf','dir_x_showerfit', 'dir_y_showerfit', 'dir_z_showerfit']).to_numpy())
            y_train = np.asarray(f["y"])
            X_train_weights = np.asarray(f["weights"])
            y_val = np.asarray(g["y"])
            X_val_weights = np.asarray(g["weights"])
            print("finished opening hdf5...")
    return X_train_scaled, X_val_scaled, y_train, X_train_weights, y_val, X_val_weights

X_train_scaled, X_val_scaled, y_train, X_train_weights, y_val, X_val_weights = get_dataset()


#tuner = tune_module.run_hyperparameter_search(X_train_scaled, y_train, X_val_scaled, y_val, X_train_weights, X_val_weights, "tune_test")
#tuner = stm.run_train(X_train_scaled, y_train, X_val_scaled, y_val, X_train_weights, X_val_weights)#, "tune_test")
tuner = kt.HPO(X_train_scaled, y_train, X_val_scaled, y_val, X_train_weights, X_val_weights)#, "tune_test")
print(tuner.get_model_summary())