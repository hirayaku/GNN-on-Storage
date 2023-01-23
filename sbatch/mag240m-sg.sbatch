#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=784G
#SBATCH --time=24:00:00
#SBATCH --output=traces/mag240m-sg.log
#SBATCH --open-mode=append

source $HOME/.bashrc
conda activate gnn
cd $HOME/proj/GNNoS
mkdir -p traces/$DATASET
date

set -eu

# helpers = trainer
srun --output=traces/$DATASET/$MODEL.NS.b$BSIZE2.out --open-mode=append \
    python3 -u trainer_ns.py --dataset $DATASET --root $ROOT --model $MODEL --comment "$COMMENT" \
    --bsize2 $BSIZE2 --bsize3 $BSIZE3 --num-hidden $HIDDEN --n-layers 3 --fanout 15,10,5 --test-fanout 30,20,10 --lr $LR --lr-decay $LRD \
    --n-epochs 25 --log-every 500 --eval-every 1 --num-workers 20 --runs $RUNS
    # --bsize2 $BSIZE2 --bsize3 $BSIZE2 --mlp --num-hidden $HIDDEN --n-layers 2 --fanout 25,15 --test-fanout 25,15 --lr $LR --lr-decay $LRD \

