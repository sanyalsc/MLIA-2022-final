#!/bin/bash
#SBATCH -A mlia
#SBATCH -p gpu
#SBATCH --gres=gpu
#SBATCH -c 1
#SBATCH -t 00:10:00

REPO_ROOT=/scratch/ejg8qa/MLIA-2022-final

module load singularity pytorch/1.12.0
singularity run $CONTAINERDIR/pytorch-1.12.0.sif -m pip install $REPO_ROOT

singularity run $CONTAINERDIR/pytorch-1.12.0.sif \
$REPO_ROOT/src/swin/MLIA-main.py --train \
--net-cfg $REPO_ROOT/configs/basic.json \
--input /scratch/ejg8qa/data