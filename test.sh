#!/bin/bash
#------- qsub option -------
#PBS -q gpu
#PBS -A ML4GW
#PBS -l elapstim_req=01:00:00
#PBS -j o
#PBS -o log/neuralnet/test.log

#------- Program execution -------
set -x
module load cuda/12.1.0
cd $PBS_O_WORKDIR
apptainer exec --nv --bind `pwd` dl4longcbc.sif ./test.py\
--outdir=data/model/firstattmpt_mdc_largesnr/20250725_084820/test_noise/\
--modeldir=data/model/firstattmpt_mdc_largesnr/20250725_084820/\
--datadir=data/dataset_250717/test/\
--ndata=0\
--batchsize=500
