#!/bin/zsh

train_path=$1
val_path=$2
name=$3

python job_HPO.py -tp $train_path -vp $val_path -n $name