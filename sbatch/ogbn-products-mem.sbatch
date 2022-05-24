#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --exclusive
#SBATCH --mem=32GB
#SBATCH --exclude=vcuda-4

cd $HOME/proj/GNN-on-Storage/baseline/graphsage
srun $CONDA_PREFIX/bin/time -v bash run-single.sh --num-epochs 1 --dataset ogbn-products

