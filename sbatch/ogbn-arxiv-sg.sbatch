#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=traces/arxiv.log
#SBATCH --open-mode=append

source $HOME/.bashrc
conda activate gnn
cd $HOME/proj/GNNoS
mkdir -p traces/arxiv
date

set -eu

# helpers = trainers
srun --output=traces/arxiv/sg.b$BSIZE2.out --open-mode=append \
    python3 -u trainer_sg.py --dataset ogbn-arxiv --root $DATASETS/baseline --comment "$COMMENT" \
    --bsize2 $BSIZE2 --fanout 15,10,5 --test-fanout 50,50,50 --lr 1e-3 --lr-decay $LRD \
    --wt-decay 5e-4 --n-epochs 100 --log-every 100 --eval-every 2

