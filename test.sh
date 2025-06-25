#!/bin/bash
#------- qsub option -------
#PBS -q gpu
#PBS -A ML4GW
#PBS -l elapstim_req=01:00:00
#PBS -v OMP_NUM_THREADS=24
#PBS -o log/out_test.out
#PBS -e log/err_test.err
#PBS -m e
#PBS -M yamamoto.s.takahiro@resceu.s.u-tokyo.ac.jp

#------- Program execution -------
module load python
module load cudnn

cd $PBS_O_WORKDIR
./test.py --outdir=cbc \
--datadir=data/250106dataset/test/ \
--ndata=10000 \
--experiment_name=first_attempt \
--run_id=8a7542441af8450f8dd523fbc7098c98 \
--batchsize=500
