import matplotlib.pyplot as plt
import optuna_tuner as ot
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '../src/dnn_energy_estimate/training/')
from data_set import get_dataset


X_train_scaled, X_val_scaled, y_train, X_train_weights, y_val, X_val_weights = get_dataset("../../")
study_name = "search_20230403"
study = ot.HPO(X_train_scaled, y_train, X_val_scaled, y_val, X_train_weights, X_val_weights, study_name = study_name)
print(study.trials_dataframe().sort_values(["value"]).head())
