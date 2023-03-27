import os, sys
import glob
import subprocess


#plot_files = glob.glob("pred_y/*")
plot_files = glob.glob("pred_y/32_32_32_32_32_32_32_32_32_32_32_32*")
python_script = 'python plot_corr.py -n '
i = 0
tot = len(plot_files)
print(plot_files)

for f in plot_files:
    i += 1
    print(i/ tot)
    print('Doing : ' +python_script + f)
    subprocess.run([python_script + f], shell=True)
    print('Done!')
