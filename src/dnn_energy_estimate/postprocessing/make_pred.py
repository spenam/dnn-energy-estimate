from tensorflow import keras
import sys
import numpy as np
import h5py
import os
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '../training/')
from data_set import get_dataset_for_pred

def make_pred(model_path, data_path, features, pred_path="./", pred_name = "default"):
    model=keras.models.load_model(model_path)
    data_scaled = get_dataset_for_pred(data_path, features)
    y_pred = model.predict(data_scaled)
    del data_scaled
    if not os.path.exists(pred_path + "/pred_y"):
        os.makedirs(pred_path +"/pred_y")
    with h5py.File(data_path, "r") as f:
        with h5py.File(pred_path + "/pred_y/" + pred_name + ".h5", "w") as g:
            g["y"]=np.log10(f["data"]["mc_energy_dst"])
            g["y_pred"]=y_pred
            g["JENERGY"]=f["data"]['JENERGY_ENERGY']
            g["JSHOWERFIT"]=f["data"]['JSHOWERFIT_ENERGY']
            g["JSTART"]=f["data"]['JSTART_LENGTH_METRES']
            g["weights"]=f["data"]["weights"]
            g["support"]=np.copy(f["support"])