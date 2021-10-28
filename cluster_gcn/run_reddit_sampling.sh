#!/bin/bash

lr="1e-2"
layers=2
psize=1500
batch_clusters=20
batch_nodes=500

python cluster_sampling.py --gpu -1 --dataset reddit-self-loop --lr $lr --weight-decay 0.0 \
  --psize $psize --batch-clusters $batch_clusters --batch-nodes $batch_nodes --n-epochs 30 --n-hidden 256 --n-layers $layers \
  --log-every 100 --self-loop --dropout 0.2 --use-val --normalize \
  --note sampling-self-loop-reddit-ly$layers-h256-pp-p$psize-b$batch_clusters-`date +"%F-%T"`

