#!/bin/bash
#PBS -l elapstim_req=01:00:00
#PBS -q gpu
#PBS -A ML4GW
#PBS -j o
#PBS -o log/mfimage_cbc/log_train
#PBS -t 0-3

# NSTART=$(($PBS_SUBREQNO * 2))
# NEND=$(($PBS_SUBREQNO * 3 + 3))

set -x
module load cuda/12.1.0
cd $PBS_O_WORKDIR
apptainer exec --bind `pwd` dl4longcbc.sif ./generate_matched_filter_image_separately.py\
	--outdir ./data/demo_250909/train/\
	--ndata 100\
    --signal

