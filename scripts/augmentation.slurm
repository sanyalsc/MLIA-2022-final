#!/bin/bash
#SBATCH -A mlia
#SBATCH -p gpu
#SBATCH --gres=gpu
#SBATCH -c 1
#SBATCH -t 00:30:00

REPO_ROOT=$1
DATA_ROOT=$2
module load singularity pytorch/1.12.0
singularity run $CONTAINERDIR/pytorch-1.12.0.sif -m pip install $REPO_ROOT

singularity run $CONTAINERDIR/pytorch-1.12.0.sif \
$REPO_ROOT/scripts/generate_augmented_data.py --input-mask \
$DATA_ROOT/train_myocardium_segmentations --input-data \
$DATA_ROOT/train_imageData --output-mask \
$DATA_ROOT/augmented_segments --output-data \
$DATA_ROOT/augmented_data \
--multiplier $3