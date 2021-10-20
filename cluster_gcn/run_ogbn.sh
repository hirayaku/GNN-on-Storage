#!/bin/bash

[[ -z $1 ]] && printf "Please provide the ogbn dataset name!\n" && exit 1

# follow parameters from the original repo
python cluster_gcn.py --gpu -1 --dataset $1 --rootdir /mnt/md0/graphs \
  --lr 1e-2 --weight-decay 0.0 --psize 1500 --batch-size 20 --n-epochs 30 \
  --n-hidden 256 --n-layers 3 --log-every 100 --dropout 0.2 \
  --note $1-ly3-h256-p1500-b20-d0_2 --use-val --normalize --feat-mmap
