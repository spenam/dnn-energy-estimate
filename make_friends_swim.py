import os,sys
import glob
import subprocess
import time

#####files =[
#####"mcv7.1.gsg_anti-elec-CC_100-500GeV.sirene.jorcarec.jsh.aanet.dst_merged.root",
#####"mcv7.1.gsg_anti-elec-CC_1-100GeV.km3sim.jorcarec.jsh.aanet.dst_merged.root",
#####"mcv7.1.gsg_anti-elec-CCHEDIS_500-10000GeV.sirene.jorcarec.jsh.aanet.dst_merged.root",
#####"mcv7.1.gsg_anti-muon-CC_100-500GeV.sirene.jorcarec.jsh.aanet.dst_merged.root",
#####"mcv7.1.gsg_anti-muon-CCHEDIS_500-10000GeV.sirene.jorcarec.jsh.aanet.dst_merged.root",
#####"mcv7.1.gsg_anti-muon-NC_100-500GeV.sirene.jorcarec.jsh.aanet.dst_merged.root",
#####"mcv7.1.gsg_anti-muon-NC_1-100GeV.km3sim.jorcarec.jsh.aanet.dst_merged.root",
#####"mcv7.1.gsg_anti-muon-NCHEDIS_500-10000GeV.sirene.jorcarec.jsh.aanet.dst_merged.root",
#####"mcv7.1.gsg_elec-CC_100-500GeV.sirene.jorcarec.jsh.aanet.dst_merged.root",
#####"mcv7.1.gsg_elec-CC_1-100GeV.km3sim.jorcarec.jsh.aanet.dst_merged.root",
#####"mcv7.1.gsg_elec-CCHEDIS_500-10000GeV.sirene.jorcarec.jsh.aanet.dst_merged.root",
#####"mcv7.1.gsg_muon-CC_100-500GeV.sirene.jorcarec.jsh.aanet.dst_merged.root",
#####"mcv7.1.gsg_muon-CCHEDIS_500-10000GeV.sirene.jorcarec.jsh.aanet.dst_merged.root",
#####"mcv7.1.gsg_muon-NC_100-500GeV.sirene.jorcarec.jsh.aanet.dst_merged.root",
#####"mcv7.1.gsg_muon-NC_1-100GeV.km3sim.jorcarec.jsh.aanet.dst_merged.root",
#####"mcv7.1.gsg_muon-NCHEDIS_500-10000GeV.sirene.jorcarec.jsh.aanet.dst_merged.root",
#####"mcv7.2.gsg_anti-elec-CC_100-500GeV.sirene.jorcarec.jsh.aanet.dst_merged.root",
#####"mcv7.2.gsg_anti-elec-CC_1-100GeV.km3sim.jorcarec.jsh.aanet.dst_merged.root",
#####"mcv7.2.gsg_anti-elec-CCHEDIS_500-10000GeV.sirene.jorcarec.jsh.aanet.dst_merged.root",
#####"mcv7.2.gsg_anti-muon-CC_100-500GeV.km3sim.jorcarec.jsh.aanet.dst_merged.root",
#####"mcv7.2.gsg_anti-muon-CC_100-500GeV.sirene.jorcarec.jsh.aanet.dst_merged.root",
#####"mcv7.2.gsg_anti-muon-CC_1-100GeV.km3sim.jorcarec.jsh.aanet.dst_merged.root",
#####"mcv7.2.gsg_anti-muon-CCHEDIS_500-10000GeV.sirene.jorcarec.jsh.aanet.dst_merged.root",
#####"mcv7.2.gsg_anti-muon-NC_100-500GeV.sirene.jorcarec.jsh.aanet.dst_merged.root",
#####"mcv7.2.gsg_anti-muon-NC_1-100GeV.km3sim.jorcarec.jsh.aanet.dst_merged.root",
#####"mcv7.2.gsg_anti-muon-NCHEDIS_500-10000GeV.sirene.jorcarec.jsh.aanet.dst_merged.root",
#####"mcv7.2.gsg_anti-tau-CC_100-500GeV.sirene.jorcarec.jsh.aanet.dst_merged.root",
#####"mcv7.2.gsg_anti-tau-CC_3-100GeV.km3sim.jorcarec.jsh.aanet.dst_merged.root",
#####"mcv7.2.gsg_elec-CC_100-500GeV.sirene.jorcarec.jsh.aanet.dst_merged.root",
#####"mcv7.2.gsg_elec-CC_1-100GeV.km3sim.jorcarec.jsh.aanet.dst_merged.root",
#####"mcv7.2.gsg_elec-CCHEDIS_500-10000GeV.sirene.jorcarec.jsh.aanet.dst_merged.root",
#####"mcv7.2.gsg_muon-CC_100-500GeV.km3sim.jorcarec.jsh.aanet.dst_merged.root",
#####"mcv7.2.gsg_muon-CC_100-500GeV.sirene.jorcarec.jsh.aanet.dst_merged.root",
#####"mcv7.2.gsg_muon-CC_1-100GeV.km3sim.jorcarec.jsh.aanet.dst_merged.root",
#####"mcv7.2.gsg_muon-CCHEDIS_500-10000GeV.sirene.jorcarec.jsh.aanet.dst_merged.root",
#####"mcv7.2.gsg_muon-NC_100-500GeV.sirene.jorcarec.jsh.aanet.dst_merged.root",
#####"mcv7.2.gsg_muon-NC_1-100GeV.km3sim.jorcarec.jsh.aanet.dst_merged.root",
#####"mcv7.2.gsg_muon-NCHEDIS_500-10000GeV.sirene.jorcarec.jsh.aanet.dst_merged.root",
#####"mcv7.2.gsg_tau-CC_100-500GeV.sirene.jorcarec.jsh.aanet.dst_merged.root",
#####"mcv7.2.gsg_tau-CC_3-100GeV.km3sim.jorcarec.jsh.aanet.dst_merged.root",
#####"v7.1.mc.mupage_tuned.sirene.jorcarec.jsh.aanet.7X_8X_dst_merged.root",
#####"v7.1.mc.mupage_tuned.sirene.jorcarec.jsh.aanet.9X_10X_dst_merged.root",
#####"v7.2.mc.mupage_tuned.sirene.jorcarec.jsh.aanet.dst_merged.root",
#####]

files =[
#"mcv7.1.gsg_anti-elec-CC_100-500GeV.sirene.jorcarec.jsh.aanet.dst_merged.root", 
"mcv7.1.gsg_anti-elec-CC_1-100GeV.km3sim.jorcarec.jsh.aanet.dst_merged.root", #
"mcv7.1.gsg_anti-elec-CCHEDIS_500-10000GeV.sirene.jorcarec.jsh.aanet.dst_merged.root", #
#"mcv7.1.gsg_anti-muon-CC_100-500GeV.sirene.jorcarec.jsh.aanet.dst_merged.root",
"mcv7.1.gsg_anti-muon-CCHEDIS_500-10000GeV.sirene.jorcarec.jsh.aanet.dst_merged.root", #
"mcv7.1.gsg_anti-muon-NC_100-500GeV.sirene.jorcarec.jsh.aanet.dst_merged.root", #
"mcv7.1.gsg_anti-muon-NC_1-100GeV.km3sim.jorcarec.jsh.aanet.dst_merged.root", #
"mcv7.1.gsg_anti-muon-NCHEDIS_500-10000GeV.sirene.jorcarec.jsh.aanet.dst_merged.root", #
#"mcv7.1.gsg_elec-CC_100-500GeV.sirene.jorcarec.jsh.aanet.dst_merged.root",
"mcv7.1.gsg_elec-CC_1-100GeV.km3sim.jorcarec.jsh.aanet.dst_merged.root", #
"mcv7.1.gsg_elec-CCHEDIS_500-10000GeV.sirene.jorcarec.jsh.aanet.dst_merged.root", #
"mcv7.1.gsg_muon-CC_100-500GeV.sirene.jorcarec.jsh.aanet.dst_merged.root",#
#"mcv7.1.gsg_muon-CCHEDIS_500-10000GeV.sirene.jorcarec.jsh.aanet.dst_merged.root",
"mcv7.1.gsg_muon-NC_100-500GeV.sirene.jorcarec.jsh.aanet.dst_merged.root", #
"mcv7.1.gsg_muon-NC_1-100GeV.km3sim.jorcarec.jsh.aanet.dst_merged.root", #
"mcv7.1.gsg_muon-NCHEDIS_500-10000GeV.sirene.jorcarec.jsh.aanet.dst_merged.root", #
"mcv7.2.gsg_anti-elec-CC_100-500GeV.km3sim.jorcarec.jsh.aanet.dst_merged.root", #
"mcv7.2.gsg_anti-elec-CC_1-100GeV.km3sim.jorcarec.jsh.aanet.dst_merged.root", #
"mcv7.2.gsg_anti-elec-CCHEDIS_500-10000GeV.sirene.jorcarec.jsh.aanet.dst_merged.root", #
"mcv7.2.gsg_anti-muon-CC_100-500GeV.km3sim.jorcarec.jsh.aanet.dst_merged.root", #
#"mcv7.2.gsg_anti-muon-CC_100-500GeV.sirene.jorcarec.jsh.aanet.dst_merged.root",
"mcv7.2.gsg_anti-muon-CC_1-100GeV.km3sim.jorcarec.jsh.aanet.dst_merged.root", #
"mcv7.2.gsg_anti-muon-CCHEDIS_500-10000GeV.sirene.jorcarec.jsh.aanet.dst_merged.root", #
"mcv7.2.gsg_anti-muon-NC_100-500GeV.sirene.jorcarec.jsh.aanet.dst_merged.root", #
"mcv7.2.gsg_anti-muon-NC_1-100GeV.km3sim.jorcarec.jsh.aanet.dst_merged.root", #
"mcv7.2.gsg_anti-muon-NCHEDIS_500-10000GeV.sirene.jorcarec.jsh.aanet.dst_merged.root", #
"mcv7.2.gsg_anti-tau-CC_100-500GeV.sirene.jorcarec.jsh.aanet.dst_merged.root", #
"mcv7.2.gsg_anti-tau-CC_3-100GeV.km3sim.jorcarec.jsh.aanet.dst_merged.root", #
#"mcv7.2.gsg_elec-CC_100-500GeV.sirene.jorcarec.jsh.aanet.dst_merged.root",
"mcv7.2.gsg_elec-CC_100-500GeV.km3sim.jorcarec.jsh.aanet.dst_merged.root", #
"mcv7.2.gsg_elec-CC_1-100GeV.km3sim.jorcarec.jsh.aanet.dst_merged.root", #
"mcv7.2.gsg_elec-CCHEDIS_500-10000GeV.sirene.jorcarec.jsh.aanet.dst_merged.root", #
"mcv7.2.gsg_muon-CC_100-500GeV.km3sim.jorcarec.jsh.aanet.dst_merged.root", #
#"mcv7.2.gsg_muon-CC_100-500GeV.sirene.jorcarec.jsh.aanet.dst_merged.root",
"mcv7.2.gsg_muon-CC_1-100GeV.km3sim.jorcarec.jsh.aanet.dst_merged.root", #
"mcv7.2.gsg_muon-CCHEDIS_500-10000GeV.sirene.jorcarec.jsh.aanet.dst_merged.root", #
"mcv7.2.gsg_muon-NC_100-500GeV.sirene.jorcarec.jsh.aanet.dst_merged.root", #
"mcv7.2.gsg_muon-NC_1-100GeV.km3sim.jorcarec.jsh.aanet.dst_merged.root", #
"mcv7.2.gsg_muon-NCHEDIS_500-10000GeV.sirene.jorcarec.jsh.aanet.dst_merged.root", #
"mcv7.2.gsg_tau-CC_100-500GeV.sirene.jorcarec.jsh.aanet.dst_merged.root", #
    "mcv7.2.gsg_tau-CC_3-100GeV.km3sim.jorcarec.jsh.aanet.dst_merged.root", #
    "v7.1.mc.mupage_tuned.sirene.jorcarec.jsh.aanet.7X_8X_dst_merged.root", #
    "v7.1.mc.mupage_tuned.sirene.jorcarec.jsh.aanet.9X_10X_dst_merged.root", #
    "v7.2.mc.mupage_tuned.sirene.jorcarec.jsh.aanet.dst_merged.root", #
]

#files = [
#    "v7.1.mc.mupage_tuned.sirene.jorcarec.jsh.aanet.7X_8X_dst_merged.root",
#    "v7.1.mc.mupage_tuned.sirene.jorcarec.jsh.aanet.9X_10X_dst_merged.root",
#    "v7.2.mc.mupage_tuned.sirene.jorcarec.jsh.aanet.dst_merged.root",
#    #"datav7.1.jorcarec.jsh.aanet.100X_dst_merged.root",
#    #"datav7.1.jorcarec.jsh.aanet.110X_dst_merged.root",
#    #"datav7.1.jorcarec.jsh.aanet.7X_dst_merged.root",
#    #"datav7.1.jorcarec.jsh.aanet.80_dst_merged.root",
#    #"datav7.1.jorcarec.jsh.aanet.85_dst_merged.root",
#    #"datav7.1.jorcarec.jsh.aanet.90_dst_merged.root",
#    #"datav7.1.jorcarec.jsh.aanet.95_dst_merged.root",
#
#]

python_script = 'python make_swim_file.py -fn '
i = 0
tot = len(files)

for f in files:
    i += 1
    print('Doing file '+str(i)+' out of ' + str(tot) + ' meaning ' + str(i/tot*100) + '%')
    print('Doing : ' + python_script + f)
    start = time.time()
    subprocess.run([python_script + f], shell=True)
    end = time.time()
    ttime = str(end-start)
    print('Done! It elapsed '+ ttime + ' seconds')
