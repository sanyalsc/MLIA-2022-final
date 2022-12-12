#!/bin/bash
#SBATCH -A mlia
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH -t 35:00:00
#SBATCH --mem=64000

REPO_ROOT=$1

module load singularity pytorch/1.12.0

singularity run $CONTAINERDIR/pytorch-1.12.0.sif -m pip install $REPO_ROOT
singularity run --nv $CONTAINERDIR/pytorch-1.12.0.sif \
$REPO_ROOT/src/swin/MLIA-main.py --train \
--net-cfg $2 \
--input $3 \
--output $4
