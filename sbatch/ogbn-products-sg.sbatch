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
mkdir -p traces/products
date

set -eu

# helpers = trainers
srun --output=traces/products/sg.b$BSIZE2.out --open-mode=append \
    python3 -u trainer_sg.py --dataset ogbn-products --root $DATASETS/baseline --comment "$COMMENT" \
    --bsize2 $BSIZE2 --bsize3 1024 --fanout 15,10,5 --test-fanout 20,20,20 --lr $LR --lr-decay $LRD \
    --n-epochs 50 --log-every 100 --eval-every 1 --lr-decay $LRD --num-workers 16
    # --wt-decay 5e-4 --n-epochs 50 --log-every 100 --eval-every 1 --lr-decay $LRD --num-workers 16

