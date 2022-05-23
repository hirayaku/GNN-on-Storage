set -x
python3 train_multi_gpu.py --n-procs 4 --dataset ogbn-papers100M --rootdir ~/datasets/baseline --num-hidden 256 --num-layers 3 --fan-out 15,10,5 $@
