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
srun --output=traces/mag240m/$PART.c$PRATIO.r$REC-$RHO.h$HSIZE.t$BSIZE.p$PSIZE.b$BSIZE2.out --open-mode=append \
    python3 -u trainer.py --dataset mag240m --root $DATASETS --comment "$COMMENT" \
    --psize $PSIZE --bsize $BSIZE --hsize $HSIZE --bsize2 $BSIZE2 --popular-ratio $PRATIO \
    --mlp --num-hidden 1024 --n-layers 2 --fanout 25,15 --test-fanout 25,15 --lr $LR --n-epochs 50 --log-every 100 --eval-every 1 \
    --lr-decay $LRD --recycle $REC --rho $RHO --part $PART --num-workers 20 # --use-incep

