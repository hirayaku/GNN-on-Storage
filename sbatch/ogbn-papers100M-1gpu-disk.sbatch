#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --nodelist=tig-slurm-2
#SBATCH --output runs/slurm.ogbn-papers100M-1gpu-disk.out

cd $HOME/proj/GNN-on-Storage/baseline/graphsage
$CONDA_PREFIX/bin/time -v bash run-multiple.sh --num-workers 0 --num-epochs 1 --dataset ogbn-papers100M --n-procs 1 --disk-feat

