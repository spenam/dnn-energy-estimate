import os, sys
import subprocess



def do_hpo(sets):
    general_path = "/sps/km3net/users/spenamar/hackathon/tf/training_workflow/extra_inputs/"
    train_path = general_path + "for_train.h5"
    val_path = general_path + "for_val.h5"
    cwd = os.getcwd()
    batch_job_script =cwd + "/" + "job_HPO.sh"
    time = "12:00:00"
    for i in range(sets):
        study_name = "study_"+str(i)
        script_options = train_path + " "+ val_path + " " + study_name
        #print('Submitting : sbatch --partition gpu --gres=gpu:v100:1 --nodes 1 --job-name hpo --ntasks 1 --mem-per-cpu 3GB --time '+time+ ' ' + batch_job_script + ' ' + script_options)
        print('Submitting : sbatch --partition gpu --gres=gpu:v100:1 --nodes 1 --job-name hpo --ntasks 1 --mem-per-cpu 9GB --time '+time+ ' ' + batch_job_script + ' ' + script_options)
        subprocess.run(['sbatch --partition gpu --gres=gpu:v100:1 --nodes 1 --job-name hpo --ntasks 1 --mem-per-cpu 9GB --time '+time+ ' ' + batch_job_script + ' ' + script_options],shell=True)


do_hpo(20)