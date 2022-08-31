#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=traces/arxiv.metis.log
#SBATCH --open-mode=append

source $HOME/.bashrc
conda activate gnn
cd $HOME/proj/GNNoS
mkdir -p traces/arxiv
date

set -eu

# helpers = trainers
srun --output=traces/arxiv/$PART.c$PRATIO.r$REC-$RHO.h$HSIZE.t$BSIZE.p$PSIZE.b$BSIZE2.out --open-mode=append \
    python3 -u trainer.py --dataset ogbn-arxiv --root $DATASETS/baseline --comment "$COMMENT" \
    --psize $PSIZE --bsize $BSIZE --hsize $HSIZE --bsize2 $BSIZE2 --popular-ratio $PRATIO \
    --fanout 15,10,5 --test-fanout 50,50,50 --lr $LR --n-epochs 100 --log-every 100 --eval-every 2 \
    --lr-decay $LRD --wt-decay 5e-4 --part $PART --recycle $REC --rho $RHO # --use-incep

