#!/bin/bash


# python cluster_gcn.py --gpu 0 --dataset reddit-self-loop --lr 1e-2 --weight-decay 0.0 --psize 1500 --batch-size 20 \
#   --n-epochs 30 --n-hidden 128 --n-layers 1 --log-every 100 --use-pp --self-loop \
#   --note self-loop-reddit-non-sym-ly3-pp-cluster-2-2-wd-5e-4 --dropout 0.2 --use-val --normalize

lr="1e-2"
layers=3
psize=1500
batch_clusters=20

# follow parameters from the original repo
python cluster_gcn.py --gpu -1 --dataset reddit-self-loop --lr $lr --weight-decay 0.0 \
  --psize $psize --batch-size $batch_clusters --n-epochs 30 --n-hidden 256 \
  --n-layers $layers --dropout 0.2 --log-every 100 --use-pp --self-loop --use-val --normalize \
  --note gcn-self-loop-reddit-ly$layers-h256-p$psize-b$batch_clusters-`date +"%F-%T"`
