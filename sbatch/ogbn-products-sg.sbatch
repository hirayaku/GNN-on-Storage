#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=traces/products.log
#SBATCH --open-mode=append

source $HOME/.bashrc
conda activate gnn
cd $HOME/proj/GNNoS
mkdir -p traces/$DATASET
date

set -eu

srun --output=traces/$DATASET/$MODEL.NS.b$BSIZE2.out --open-mode=append \
    python3 -u trainer_ns.py --dataset ogbn-products --root $DATASETS/gnnos --model $MODEL --comment "$COMMENT" \
    --num-hidden $HIDDEN --bsize2 $BSIZE2 --bsize3 1024 --fanout 15,10,5 --test-fanout 20,20,20 \
    --lr $LR --lr-decay $LRD --wt-decay $WD --n-epochs 50 --log-every 100 --eval-every 1 --num-workers 16 \
    --runs $RUNS

