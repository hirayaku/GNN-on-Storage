#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=384G
#SBATCH --time=24:00:00
#SBATCH --output=traces/papers.log
#SBATCH --open-mode=append

source $HOME/.bashrc
conda activate gnn
cd $HOME/proj/GNNoS
mkdir -p traces/$DATASET
date

set -eu

srun --output=traces/$DATASET/$MODEL.NS.b$BSIZE2.out --open-mode=append \
    python3 -u trainer_sg.py --dataset ogbn-papers100M --root $ROOT --model $MODEL --comment "$COMMENT" \
    --num-hidden $HIDDEN --bsize2 $BSIZE2 --bsize3 $BSIZE3 --fanout $FANOUT --lr $LR --n-epochs 25 --log-every 300 --eval-every 1 \
    --dropout $DPOUT --lr-decay $LRD --lr-step $LRS --num-workers 20 --runs $RUNS
