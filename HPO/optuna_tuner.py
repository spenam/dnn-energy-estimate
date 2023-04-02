from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers, Sequential
import optuna
import joblib
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '../src/dnn_energy_estimate/training/')
import build_train as trainer


def run_train(trial, x_train, y_train, x_val, y_val, x_train_weights, x_val_weights):
    fast_config = {
        "n_layers": trial.suggest_int("n_layers",16,32,16),#[4,8,12,16,20,24]),
        "n_nodes": 16,
        "batch_size": 32,
        "batchnorm": 1,
        "lossf": "mean_squared_error",
        "activation": "ReLU",
        "drop_vals": 0.,
        "learning_rate": 1e-6,
    }
    config = {
        "n_layers": trial.suggest_int("n_layers",4,32,4),#[4,8,12,16,20,24]),
        "n_nodes": trial.suggest_int("n_nodes",16,128,16),#[16,32,64,128]),
        "batch_size": trial.suggest_int("batch_size",16,128,16),#[16,32,64,128]),
        "batchnorm": trial.suggest_int("batchnorm",0,1),#[True, False]),
        "lossf": trial.suggest_categorical("lossf",["mean_squared_error", "log_cosh"]),
        "activation": trial.suggest_categorical("activation",["PReLU", "ReLU"]),
        "drop_vals": trial.suggest_float("drop_vals",0,0.5),#[0., 0.1, 0.2, 0.3, 0.4, 0.5]),
        "learning_rate": trial.suggest_float("learning_rate",1e-6,1e-3,log=True),#[1e-3, 1e-4, 1e-5, 1e-6]),
    }
    data = {
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "x_train_weights": x_train_weights,
        "x_val_weights": x_val_weights,
    }
    train = trainer.TrainNN(fast_config, data)
    return train.step()
def HPO(x_train, y_train, x_val, y_val, x_train_weights, x_val_weights, study_name="default"):
    n_trials = 3
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: run_train(trial, x_train, y_train, x_val, y_val, x_train_weights, x_val_weights), n_trials=n_trials)
    print("Best config: ", study.best_params)
    joblib.dump(study, study_name+".pkl")
    return study