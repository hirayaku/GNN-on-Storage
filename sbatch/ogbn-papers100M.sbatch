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
srun --output=traces/papers100M/$PART.c$PRATIO.r$REC-$RHO.h$HSIZE.t$BSIZE.p$PSIZE.b$BSIZE2.out --open-mode=append \
    python3 -u trainer.py --dataset ogbn-papers100M --root $DATASETS/baseline --comment "$COMMENT" \
    --psize $PSIZE --bsize $BSIZE --hsize $HSIZE --bsize2 $BSIZE2 --popular-ratio $PRATIO \
    --fanout 15,10,5 --lr $LR --n-epochs 25 --log-every 100 --eval-every 1 \
    --lr-decay $LRD --part $PART --recycle $REC --rho $RHO # --use-incep

