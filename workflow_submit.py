import os
import sys
import subprocess

# No likelihood for now
JGandalfcols = [
    "JENERGY_ENERGY",
    "JSTART_LENGTH_METRES",
    "dir_x_gandalf",
    "dir_y_gandalf",
    "dir_z_gandalf",
]  # Features to keep
JShowerfitcols = [
    "JSHOWERFIT_LENGTH_METRES",
    "JSHOWERFIT_ENERGY",
    "dir_x_showerfit",
    "dir_y_showerfit",
    "dir_z_showerfit",
]
likelihoods = ["lik_first_JENERGY", "lik_first_JSHOWERFIT"]
Triggercols = ["trig_hits", "trig_doms", "trig_lines"]
MCcols = ["mc_energy_dst", "mc_pos_x", "mc_pos_y", "mc_pos_z", "weights"]


GNNcols = []  # No GNN for now

trainingfeatures = JGandalfcols + JShowerfitcols + Triggercols + GNNcols
allfeatures = MCcols + trainingfeatures
print(allfeatures)
n_features = len(trainingfeatures)


architecture_size = ["16", "32"]
architecture_layers = ["6", "12"]
lr = ["0.00001", "0.000001"]  # , "0.000005"]
loss = ["log_cosh", "mean_squared_error"]
# 0 for training without likelihood and 1 for training with likelihood
lkl = ["0", "5"]
architecture_size = ["32"]
architecture_layers = ["12"]
lkl = ["5"]  # 0 for training without likelihood and 1 for training with likelihood
# architecture_size = ["32"]
# architecture_layers = ["12"]
# lkl = ["5"] # 0 for training without likelihood and 5 for training with likelihood
# architecture_size = ["4"]
# architecture_layers = ["1"]
# lr = ["0.00001"]#, "0.000001"]#, "0.000005"]


def do_samples():
    cwd = os.getcwd()
    batch_job_script = cwd + "/" + "make_train_val.sh"
    script_options = ""
    time = "00:30:00"
    print(
        "Submitting : sbatch --partition htc --nodes 1 --job-name make_training_samples --ntasks 1 --mem-per-cpu 64GB --time "
        + time
        + " "
        + batch_job_script
        + " "
        + script_options
    )
    subprocess.run(
        [
            "sbatch --partition htc --nodes 1 --job-name make_training_samples --ntasks 1 --mem-per-cpu 64GB --time "
            + time
            + " "
            + batch_job_script
            + " "
            + script_options
        ],
        shell=True,
    )


def do_training(online=1):
    cwd = os.getcwd()
    batch_job_script = cwd + "/" + "train_model_new.sh"
    time = "12:00:00"
    time = "3:00:00"
    memory = "8GB"
    for ll, _ in enumerate(lkl):
        for l, _ in enumerate(loss):
            for k, _ in enumerate(architecture_layers):
                for i, _ in enumerate(architecture_size):
                    for j, item in enumerate(lr):
                        if lkl[ll] == "5":
                            trainingfeatures = (
                                JGandalfcols
                                + JShowerfitcols
                                + Triggercols
                                + GNNcols
                                + likelihoods
                            )
                        else:
                            trainingfeatures = (
                                JGandalfcols + JShowerfitcols + Triggercols + GNNcols
                            )
                        allfeatures = MCcols + trainingfeatures
                        n_features = len(trainingfeatures)

                        script_options = (
                            architecture_layers[k]
                            + " "
                            + architecture_size[i]
                            + " "
                            + item
                            + " "
                            + str(n_features)
                            + " "
                            + loss[l]
                            + " "
                            + lkl[ll]
                        )
                        if online == 1:
                            print(
                                "Submitting : sbatch --partition htc --nodes 1 --job-name DNN_training --ntasks 1 --mem-per-cpu "
                                + memory
                                + " --time "
                                + time
                                + " "
                                + batch_job_script
                                + " "
                                + script_options
                            )
                            subprocess.run(
                                [
                                    "sbatch --partition htc --nodes 1 --job-name DNN_training --ntasks 1 --mem-per-cpu "
                                    + memory
                                    + " --time "
                                    + time
                                    + " "
                                    + batch_job_script
                                    + " "
                                    + script_options
                                ],
                                shell=True,
                            )
                        else:
                            subprocess.run(
                                [batch_job_script + " " + script_options], shell=True
                            )


# do_samples()
do_training(online=0)
