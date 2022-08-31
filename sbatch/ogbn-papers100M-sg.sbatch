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
mkdir -p traces/papers100M
date

set -eu

# helpers = trainers
srun --output=traces/papers100M/sg.b$BSIZE2.out --open-mode=append \
    python3 -u trainer_sg.py --dataset ogbn-papers100M --root $DATASETS/baseline --comment "$COMMENT" \
    --bsize2 $BSIZE2 --fanout 15,10,5 --lr $LR --n-epochs 25 --log-every 100 --eval-every 1 \
    --lr-decay $LRD --num-workers 20 