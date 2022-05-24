#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --nodelist=tig-slurm-2
#SBATCH --output runs/slurm.mag240m-mem.out

cd $HOME/proj/GNN-on-Storage/baseline/graphsage
$CONDA_PREFIX/bin/time -v bash run-single.sh --dataset mag240m --num-epochs 50 --eval-every 50 # don't validate

