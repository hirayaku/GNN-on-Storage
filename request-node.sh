#!/bin/bash

[[ -z $GPUS ]] && GPUS=1
[[ -z $CPUS ]] && CPUS=32
[[ -z $MEM ]] && MEM=64G
[[ -n $NODE ]] && NODE="--nodelist=$NODE"

OPT_GPUS="--gres=gpu:$GPUS"
OPT_CPUS="-c $CPUS"
OPT_MEM="--mem $MEM"

set -x
srun $OPT_GPUS -N 1 -n 1 $OPT_CPUS $OPT_MEM $NODE --time 24:00:00 --pty bash

