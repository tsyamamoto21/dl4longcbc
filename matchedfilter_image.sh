#!/bin/bash
#PBS -l elapstim_req=02:00:00
#PBS -l cpunum_job=48
#PBS -q gpu
#PBS -A ML4GW
#PBS -j o
#PBS -o log/mfimage_noise/log
#PBS -t 0-3

INJECTION_FILE=data/mdc/ds1_val/injection.hdf
FOREGROUND_FILE=data/mdc/ds1_val/foreground.hdf
NSTART=$(($PBS_SUBREQNO))
NEND=$(($PBS_SUBREQNO + 1))

set -x
module load cuda/12.1.0
cd $PBS_O_WORKDIR
pwd
apptainer exec --bind `pwd` dl4longcbc.sif ./use_mdc_generate_matchedfilter_image.py\
	--outdir=data/dataset_250625/validate/noise\
	--foreground=$FOREGROUND_FILE\
	--injection=$INJECTION_FILE\
    --nstart=$NSTART\
    --nend=$NEND\
    --offevent

