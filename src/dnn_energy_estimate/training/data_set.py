import h5py
import tensorflow as tf
import numpy as np
import pandas as pd

def get_dataset(fpath_train: str, fpath_val: str, features):
    with h5py.File(fpath_train, "r") as f:
        with h5py.File(fpath_val, "r") as g:
            print("Opening hdf5...")
            X_train_scaled = tf.convert_to_tensor(pd.DataFrame({key: f["X"][key][:] for key in features}).to_numpy())
            X_val_scaled = tf.convert_to_tensor(pd.DataFrame({key: g["X"][key][:] for key in features}).to_numpy())
            y_train = np.asarray(f["y"])
            X_train_weights = np.asarray(f["weights"])
            y_val = np.asarray(g["y"])
            X_val_weights = np.asarray(g["weights"])
            print("finished opening hdf5...")
    return X_train_scaled, X_val_scaled, y_train, X_train_weights, y_val, X_val_weights

def get_dataset_for_pred(fpath: str, features):
    with h5py.File(fpath, "r") as f:
        print("Opening hdf5...")
        fulldata = f['data']
        data_scaled = tf.convert_to_tensor(pd.DataFrame({key: fulldata[key][:] for key in features}).to_numpy())
        return data_scaled
