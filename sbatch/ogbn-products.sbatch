#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=traces/ogbn-products.log
#SBATCH --open-mode=append

source $HOME/.bashrc
conda activate gnn
cd $HOME/proj/GNNoS
mkdir -p traces/$DATASET
date

set -eu

srun --output=traces/$DATASET/$PART.c$PRATIO.r$REC-$RHO.t$BSIZE.p$PSIZE.b$BSIZE2.out --open-mode=append \
    python3 -u trainer.py --dataset ogbn-products --root $ROOT --model $MODEL --comment "$COMMENT" \
    --psize $PSIZE --bsize $BSIZE --bsize2 $BSIZE2 --bsize3 1024 --popular-ratio $PRATIO \
    --lr $LR --lr-decay $LRD --wt-decay $WD --part $PART --recycle $REC --rho $RHO \
    --fanout 15,10,5 --test-fanout 20,20,20 --n-epochs 50 --log-every 100 --eval-every 1 --num-workers 16
