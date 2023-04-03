import matplotlib.pyplot as plt
import optuna_tuner as ot
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '../src/dnn_energy_estimate/training/')
#from data_set import get_dataset
features_1 = ['JSHOWERFIT_ENERGY', 'JENERGY_ENERGY', 'lik_first_JENERGY', 'lik_first_JSHOWERFIT', 'trig_hits', 'trig_doms', 'trig_lines', 'JSTART_LENGTH_METRES', 'dir_x_gandalf', 'dir_y_gandalf', 'dir_z_gandalf','dir_x_showerfit', 'dir_y_showerfit', 'dir_z_showerfit']
features_2 = ['JSTART_LENGTH_METRES', 'JSHOWERFIT_ENERGY', 'JENERGY_ENERGY', 'trig_hits', 'trig_doms', 'trig_lines', 'dir_x_gandalf', 'dir_y_gandalf', 'dir_z_gandalf', 'dir_x_showerfit', 'dir_y_showerfit', 'dir_z_showerfit', 'lik_first_JENERGY', 'lik_first_JSHOWERFIT',  'trackscore', 'muonscore']
features_3 = ['JSTART_LENGTH_METRES', 'JSHOWERFIT_ENERGY', 'JENERGY_ENERGY', 'trig_hits', 'trig_doms', 'trig_lines', 'dir_x_gandalf', 'dir_y_gandalf', 'dir_z_gandalf', 'dir_x_showerfit', 'dir_y_showerfit', 'dir_z_showerfit', 'lik_first_JENERGY', 'lik_first_JSHOWERFIT', 'pos_x_gandalf', 'pos_y_gandalf', 'pos_z_gandalf', 'pos_x_showerfit', 'pos_y_showerfit', 'pos_z_showerfit', 'trackscore', 'muonscore']

train_path = "../../for_train.h5"
val_path = "../../for_val.h5"
#X_train_scaled, X_val_scaled, y_train, X_train_weights, y_val, X_val_weights = get_dataset("../../", features)
#study_name = "search_20230403"
study_name = "fast_test"
#study = ot.HPO(X_train_scaled, y_train, X_val_scaled, y_val, X_train_weights, X_val_weights, study_name = study_name)
study = ot.HPO(train_path, val_path, study_name = study_name)
#print(study.trials_dataframe().sort_values(["value"]).head())
