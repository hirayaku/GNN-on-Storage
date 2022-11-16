#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=300G
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
    python3 -u trainer_hb.py --dataset ogbn-papers100M --root $ROOT --model $MODEL --comment "$COMMENT" \
    --num-hidden $HIDDEN --bsize2 $BSIZE2 --bsize3 $BSIZE3 --fanout $FANOUT --test-fanout 20,20,20 --lr $LR --n-epochs 25 --log-every 20 --eval-every 1 \
    --dropout $DPOUT --lr-decay $LRD --lr-step $LRS --num-workers 8 --runs $RUNS --io-threads 64 \
    --part $PART --psize $PSIZE --bsize $BSIZE --popular-ratio $PRATIO --recycle $REC --rho $RHO

