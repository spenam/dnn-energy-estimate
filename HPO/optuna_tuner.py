import optuna
import joblib
import sys
import os
import matplotlib.pyplot as plt
import glob
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '../src/dnn_energy_estimate/')
from training.data_set import get_dataset
import training.build_train as trainer
from postprocessing.test_pred import make_test
from plotting.plot_corr import do_plots


def _config_to_str(dic):
        return "-".join([f"{k}_{v}" for k, v in dic.items()])


#def run_train(trial, x_train, y_train, x_val, y_val, x_train_weights, x_val_weights, study_name):
def run_train(trial, train_path, val_path, study_name):
    fts ={
    "fts_1" : ['JSHOWERFIT_ENERGY', 'JENERGY_ENERGY', 'lik_first_JENERGY', 'lik_first_JSHOWERFIT', 'trig_hits', 'trig_doms', 'trig_lines', 'JSTART_LENGTH_METRES', 'dir_x_gandalf', 'dir_y_gandalf', 'dir_z_gandalf','dir_x_showerfit', 'dir_y_showerfit', 'dir_z_showerfit'],
    "fts_2" : ['JSTART_LENGTH_METRES', 'JSHOWERFIT_ENERGY', 'JENERGY_ENERGY', 'trig_hits', 'trig_doms', 'trig_lines', 'dir_x_gandalf', 'dir_y_gandalf', 'dir_z_gandalf', 'dir_x_showerfit', 'dir_y_showerfit', 'dir_z_showerfit', 'lik_first_JENERGY', 'lik_first_JSHOWERFIT',  'trackscore', 'muonscore'],
    "fts_3" : ['JSTART_LENGTH_METRES', 'JSHOWERFIT_ENERGY', 'JENERGY_ENERGY', 'trig_hits', 'trig_doms', 'trig_lines', 'dir_x_gandalf', 'dir_y_gandalf', 'dir_z_gandalf', 'dir_x_showerfit', 'dir_y_showerfit', 'dir_z_showerfit', 'lik_first_JENERGY', 'lik_first_JSHOWERFIT', 'pos_x_gandalf', 'pos_y_gandalf', 'pos_z_gandalf', 'pos_x_showerfit', 'pos_y_showerfit', 'pos_z_showerfit', 'trackscore', 'muonscore'],
    }
    #fast_config = {
    #    "n_layers": trial.suggest_int("n_layers",32,64,8),#[4,8,12,16,20,24]),
    #    "n_nodes": trial.suggest_int("n_nodes",32,64,8),
    #    "batch_size": 2048,#32,
    #    "batchnorm": 1,
    #    "lossf": "mean_squared_error",
    #    "activation": "ReLU",
    #    "drop_vals": 0.,
    #    "learning_rate": 1e-6,
    #    "features" : "fts_1",
    #}
    config = {
        "n_layers": trial.suggest_int("n_layers",16,32,step=16),#[4,8,12,16,20,24]),
        "n_nodes": trial.suggest_int("n_nodes",32,64,step=32),#[16,32,64,128]),
        #"n_layers": 12,
        #"n_nodes": 32,#[16,32,64,128]),
        #"batch_size": trial.suggest_int("batch_size",16,128,16),#[16,32,64,128]),
        #"batch_size": 32,
        #"batchnorm": trial.suggest_int("batchnorm",0,1),#[True, False]),
        "lossf": trial.suggest_categorical("lossf",["mean_squared_error", "log_cosh"]),
        #"activation": trial.suggest_categorical("activation",["PReLU", "ReLU"]),
        #"activation": "PReLU",
        #"drop_vals": trial.suggest_float("drop_vals",0,0.1,step=0.1),#[0., 0.1, 0.2, 0.3, 0.4, 0.5]),
        #"drop_vals": 0.,#[0., 0.1, 0.2, 0.3, 0.4, 0.5]),
        #"learning_rate": trial.suggest_float("learning_rate",1e-7,1e-5,log=True),#[1e-3, 1e-4, 1e-5, 1e-6]),
        #"learning_rate": 1e-5,#[1e-3, 1e-4, 1e-5, 1e-6]),
        "l2": trial.suggest_float("l2", 1e-3,1e-2, log=True),#, step=1e-3),
        "features" : trial.suggest_categorical("features",["fts_1", "fts_2", "fts_3"]),
        #"features" : "fts_1"
    }
    #config = fast_config
    x_train, x_val, y_train, x_train_weights, y_val, x_val_weights = get_dataset(train_path, val_path, fts[config["features"]])
    data = {
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "x_train_weights": x_train_weights,
        "x_val_weights": x_val_weights,
    }
    train = trainer.TrainNN(config, data, study_name)
    #train = trainer.TrainNN(config, data, study_name)
    value = train.step()
    return value

def HPO(train_path, val_path, study_name="default"):
    n_trials = 2 #20 #100
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: run_train(trial, train_path, val_path, study_name), n_trials=n_trials)
    orig_stdout = sys.stdout
    which_feature = study.best_params["features"]
    abspath = None
    with open(study_name + "/result.txt","w") as f:
        sys.stdout = f
        print("Best config: ", study.best_params)
        all_tests = glob.glob(study_name + "/*")
        for item in all_tests:
             if (item.find(_config_to_str(study.best_params)) != -1) & (item.find(".h5") != -1):
                  abspath = os.path.abspath(item)
                  print("path of best item ",abspath)
                  break
        print(study.trials_dataframe().sort_values(["value"]).head(10))
    sys.stdout.close()
    sys.stdout=orig_stdout 
    print("Best config: ", study.best_params)
    print(study.trials_dataframe().sort_values(["value"]).head(10))
    joblib.dump(study, study_name + "/" + study_name+".pkl")
    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_image(study_name + "/optm_history.pdf")
    fig.write_image(study_name + "/optm_history.png")
    fig = optuna.visualization.plot_parallel_coordinate(study)
    fig.write_image(study_name + "/parallel_coordinate.pdf")
    fig.write_image(study_name + "/parallel_coordinate.png")
    fig = optuna.visualization.plot_contour(study)
    fig.write_image(study_name + "/contour.pdf")
    fig.write_image(study_name + "/contour.png")
    print("path of best item ",abspath)
    print("Doing prediction...")
    study_abspath = os.path.abspath(study_name)
    pred_path = make_test(abspath, which_feature, study_abspath)
    print("Done with prediction!!!")
    print("Doing plotting...")
    do_plots(pred_path, study_abspath)
    print("Done with plotting!!!")
    


    return study