import matplotlib.pyplot as plt
import optuna_tuner as ot
import sys
import argparse
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '../src/dnn_energy_estimate/training/')

parser = argparse.ArgumentParser(description="HPO arguments")
parser.add_argument(
    "-tp", "--train_path", type = str, help="path for training data", required=True
)
parser.add_argument(
    "-vp", "--val_path", type = str, help="path for validation data", required=True
)
parser.add_argument(
    "-n", "--name", type = str, help="name of study", required=True
)
args = parser.parse_args()
train_path = args.train_path
val_path = args.val_path
study_name = args.name
study = ot.HPO(train_path, val_path, study_name = study_name)