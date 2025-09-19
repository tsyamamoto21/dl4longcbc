#!/bin/bash
#------- qsub option -------
#PBS -q gpu
#PBS -A ML4GW
#PBS -l elapstim_req=24:00:00
#PBS -j o
#PBS -o log/neuralnet/train4.log

#------- Program execution -------
set -x
module load cuda/12.1.0
cd $PBS_O_WORKDIR
apptainer exec --nv --bind `pwd` dl4longcbc.sif ./train.py --dirname gmn0.5_ksize5-13_channels64_relu_AdamW
