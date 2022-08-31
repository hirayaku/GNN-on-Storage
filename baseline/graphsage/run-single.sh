set -x
python3 train_single.py --data-cpu --gpu 0 --dataset ogbn-papers100M --rootdir ~/datasets/baseline --num-hidden 256 --num-layers 3 --fan-out 15,10,5 $@
