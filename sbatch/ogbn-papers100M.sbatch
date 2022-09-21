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

srun --output=traces/$DATASET/$PART.$MODEL.c$PRATIO.r$REC-$RHO.t$BSIZE.p$PSIZE.b$BSIZE2.out --open-mode=append \
    python3 -u trainer.py --dataset ogbn-papers100M --root $ROOT --model $MODEL --comment "$COMMENT" \
    --num-hidden $HIDDEN --bsize2 $BSIZE2 --fanout 15,10,5 --lr $LR --n-epochs 25 --log-every 100 --eval-every 1 \
    --dropout $DPOUT --lr-decay $LRD --lr-step $LRS --num-workers 20 --use-incep --runs $RUNS \
    --part $PART --psize $PSIZE --bsize $BSIZE --popular-ratio $PRATIO --recycle $REC --rho $RHO

