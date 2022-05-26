#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --nodelist=tig-slurm-2
#SBATCH --output runs/slurm.ogbn-papers100M-disk.out

cd $HOME/proj/GNN-on-Storage/baseline/graphsage
$CONDA_PREFIX/bin/time -v bash run-single.sh --num-workers 0 --num-epochs 50 --eval-every 50 --disk-feat

