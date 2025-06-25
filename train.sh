#!/bin/bash
#------- qsub option -------
#PBS -q gpu
#PBS -A ML4GW
#PBS -l elapstim_req=01:00:00
#PBS -v OMP_NUM_THREADS=24
#PBS -o log/train.out
#PBS -e log/train.out

#------- Program execution -------
module load python
module load cudnn

cd $PBS_O_WORKDIR
./train.py
