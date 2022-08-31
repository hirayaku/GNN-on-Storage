#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --time=24:00:00
#SBATCH --output=traces/mag240m.log
#SBATCH --open-mode=append

source $HOME/.bashrc
conda activate gnn
cd $HOME/proj/GNNoS
mkdir -p traces/mag240m
date

set -eu

# helpers = trainer
srun --output=traces/mag240m/sg.b$BSIZE2.out --open-mode=append \
    python3 -u trainer_sg.py --dataset mag240m --root $DATASETS --comment "$COMMENT" \
    --bsize2 $BSIZE2 --mlp --num-hidden 1024 --n-layers 2 --fanout 25,15 --test-fanout 25,15 --lr $LR --lr-decay $LRD \
    --n-epochs 50 --log-every 100 --eval-every 1 --num-workers 20 # --use-incep

