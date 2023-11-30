#!/bin/bash

LOGDIR=./logdir/profile
[[ -z $RUN ]] && RUN=prof
CUDA_LOG=cuda-$RUN.log
IO_LOG=io-$RUN.log
if [[ $# -gt 0 ]]; then
    timeout 10m python3 $* | tee $LOGDIR/run.out &
    timeout 10m nvidia-smi dmon -i 0 > $LOGDIR/$CUDA_LOG &
    timeout 10m dstat --part -P total -rc > $LOGDIR/$IO_LOG &
else
    echo "no script given"
fi

