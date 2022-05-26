#!/bin/bash

#SBATCH --time=00:10:00
#SBATCH --partition=gpu
#SBATCH --exclusive
#SBATCH --mem=32GB
#SBATCH --exclude=vcuda-4

cd $HOME/proj/GNN-on-Storage/baseline/graphsage
$CONDA_PREFIX/bin/time -v bash run-multiple.sh --num-epochs 1 --dataset ogbn-products --n-procs 1

