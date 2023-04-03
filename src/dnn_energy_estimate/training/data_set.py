import h5py
import tensorflow as tf
import numpy as np
import pandas as pd

def get_dataset(fpath: str, features):
    with h5py.File(fpath + "/for_train.h5", "r") as f:
        with h5py.File(fpath + "/for_val.h5", "r") as g:
            print("Opening hdf5...")
            #X_train_scaled = tf.convert_to_tensor(pd.DataFrame(f["X"]['JSHOWERFIT_ENERGY',
            #                                                              'JENERGY_ENERGY', 'lik_first_JENERGY',
            #                                                              'lik_first_JSHOWERFIT', 'trig_hits', 'trig_doms', 'trig_lines', 'JSTART_LENGTH_METRES', 'dir_x_gandalf', 'dir_y_gandalf', 'dir_z_gandalf','dir_x_showerfit', 'dir_y_showerfit', 'dir_z_showerfit']).to_numpy())
            #X_val_scaled = tf.convert_to_tensor(pd.DataFrame(g["X"]['JSHOWERFIT_ENERGY',
            #                                                              'JENERGY_ENERGY', 'lik_first_JENERGY',
            #                                                              'lik_first_JSHOWERFIT', 'trig_hits', 'trig_doms', 'trig_lines', 'JSTART_LENGTH_METRES', 'dir_x_gandalf', 'dir_y_gandalf', 'dir_z_gandalf','dir_x_showerfit', 'dir_y_showerfit', 'dir_z_showerfit']).to_numpy())
            X_train_scaled = tf.convert_to_tensor(pd.DataFrame({key: f["X"][key][:] for key in features}).to_numpy())
            X_val_scaled = tf.convert_to_tensor(pd.DataFrame({key: g["X"][key][:] for key in features}).to_numpy())
            y_train = np.asarray(f["y"])
            X_train_weights = np.asarray(f["weights"])
            y_val = np.asarray(g["y"])
            X_val_weights = np.asarray(g["weights"])
            print("finished opening hdf5...")
    return X_train_scaled, X_val_scaled, y_train, X_train_weights, y_val, X_val_weights

def get_dataset_for_pred(fpath: str):
    with h5py.File(fpath, "r") as f:
        print("Opening hdf5...")
        fulldata = f['data']
        data_scaled = tf.convert_to_tensor(pd.DataFrame(fulldata['JSHOWERFIT_ENERGY',
                                                                      'JENERGY_ENERGY', 'lik_first_JENERGY',
                                                                      'lik_first_JSHOWERFIT', 'trig_hits', 'trig_doms', 'trig_lines', 'JSTART_LENGTH_METRES', 'dir_x_gandalf', 'dir_y_gandalf', 'dir_z_gandalf','dir_x_showerfit', 'dir_y_showerfit', 'dir_z_showerfit']).to_numpy())
        return data_scaled
