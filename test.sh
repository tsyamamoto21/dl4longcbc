#!/bin/bash
#------- qsub option -------
#PBS -q gpu
#PBS -A ML4GW
#PBS -l elapstim_req=00:30:00
#PBS -j o
#PBS -o log/neuralnet/test.log

#------- Program execution -------
set -x
module load cuda/12.1.0
cd $PBS_O_WORKDIR
apptainer exec --nv --bind `pwd` dl4longcbc.sif ./test.py\
    --outdir=data/model/model_250912/gmn10.0_ksize5-13_channels64_relu/test_cbc/\
    --modeldir=data/model/model_250912/gmn10.0_ksize5-13_channels64_relu/\
    --datadir=data/dataset_250911/test/\
    --ndata=10000\
    --batchsize=100\
    --cbc
