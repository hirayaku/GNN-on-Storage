#!/bin/bash

lr="1e-2"
layers=3
fanout=5,10,15
psize=1500
batch_clusters=20
batch_nodes=520

python cluster_cache_sampling.py --gpu -1 --dataset reddit-self-loop --lr $lr --weight-decay 0.0 \
  --psize $psize --batch-clusters $batch_clusters --batch-nodes $batch_nodes \
  --n-epochs 30 --n-hidden 256 --n-layers $layers --fan-out $fanout --dropout 0.2\
  --self-loop --log-every 100 --use-val --normalize \
  --note cache-self-loop-reddit-ly$layers-h256-p$psize-b$batch_clusters-`date +"%F-%T"`
