#!/bin/sh

nl=$1
ls=$2
lr=$3
nf=$4
lf=$5
lkl=$6

#python train_model_new.py -nl $nl -ls $ls -lr $lr -lf $lf -lkl $lkl
echo "Done with the training! :D"
python make_pred_new.py -nl $nl -ls $ls -lr $lr -nf $nf -lf $lf -lkl $lkl
echo "Done with the predictions! :D"
#echo "Doing performance plots! "
#python make_perf_plots.py -nl $nl -ls $ls -lr $lr -nf $nf -lf $lf -lkl $lkl
#echo "Done with the plots! :D"
