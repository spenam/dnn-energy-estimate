import pandas as pd  # data analysis package
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from functions import *
#from plotting import *
from time import time
import matplotlib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
import h5py

from ray import tune 
from ray import air 
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
#import optuna
import h5py

def get_dataset():#fpath: str):
    with h5py.File("../for_train.h5", "r") as f:
        with h5py.File("../for_val.h5", "r") as g:
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

keras_tune_kwargs = {
    "n_layers": tune.quniform(16,32,4),
    "n_nodes": tune.quniform(16,64,2),
    "learning_rate": tune.quniform(-5.0, -1.0, 0.5),  # powers of 10
    "drop_out": tune.quniform(0,0.4,0.1),
    "batchnorm": tune.choice([True,False]),
    "activation": tune.choice(["PReLU","ReLU"])
}
RANDOMSTATE=42
kfolds = 10

X_train_scaled, X_val_scaled, y_train, X_train_weights, y_val, X_val_weights = get_dataset()

#keras_tune_params = [k for k in xgb_tune_kwargs.keys() if k != 'wandb']
keras_tune_params = [k for k in keras_tune_kwargs.keys()]

def build_network(**config):
    model = Sequential()
    model.add(layers.BatchNormalization())
    for j in range(n_layers):
        model.add(layers.Dense(n_nodes))
        if batchnorm is True:
            model.add(layers.BatchNormalization())
        model.add(layers.Activation(activation))
        model.add(layers.Dropout(drop_out))
    model.add(layers.Dense(1))

def my_keras(config):

    # fix these configs to match calling convention
    # search wants to pass in floats but keras wants ints
    config['max_depth'] = int(config['max_depth']) + 2
    config['n_layers'] = int(config['n_layers']) 
    config['n_nodes'] = int(config['n_nodes']) 
    config['learning_rate'] = 10 ** config['learning_rate']
    config['drop_out'] = int(config['drop_out']) 
    config['batchnorm'] = int(config['batchnorm']) 
    lossf = "mean_squared_error"
    
    model = build_network(
        **config,
    )

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=config['learning_rate']), loss=lossf)
        
    #scores = cross_val_score(keras, df[predictors], df[response],
    #                                  scoring="mean_squared_error",
    #                                  cv=kfolds)
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
    mse = history.history["loss"][-1]
    tune.report(mse=mse)
    
    return {"mse": mse}

algo = OptunaSearch()
# ASHA
scheduler = ASHAScheduler()

NUM_SAMPLES = 10
#train_model = tune.with_resources(train_model, {"cpu": 1})

analysis = tune.Tuner(my_keras,
                    keras_tune_kwargs,                    
                    run_config=air.RunConfig(name="optuna_keras"),
                    tune_config=tune.TuneConfig(num_samples=NUM_SAMPLES),
                    #metric="mse",
                    #mode="min",
                    search_alg=algo,
                    scheduler=scheduler,
                    verbose=1,
                   )

results = analysis.fit()
print(analysis.get_results())



# results dataframe sorted by best metric
param_cols = ['config.' + k for k in keras_tune_params]
analysis_results_df = analysis.results_df[['mse', 'date', 'time_this_iter_s'] + param_cols].sort_values('mse')

# extract top row
best_config = {z: analysis_results_df.iloc[0]['config.' + z] for z in keras_tune_params}



print(analysis_results_df)
