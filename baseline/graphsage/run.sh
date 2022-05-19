# in-memory execution on CPU
#python train_single.py --gpu -1 --dataset ogbn-products --num-epochs 2

# in-disk execution on a single GPU
#python train_single.py --gpu 0 --dataset ogbn-products --rootdir ~/datasets/dgl-data/ --data-cpu --disk-feat --num-epochs 2

# in-memory execution on multi-GPU
#python train_multi_gpu.py --dataset ogbn-products --num-epochs=2 --gpu 0

#python train_multi_gpu.py --dataset ogbn-products --num-epochs=20 --gpu 0  --eval-every 5

# in-disk execution on multi-GPU
python train_multi_gpu.py --dataset ogbn-products --num-epochs=20 --gpu 0 --disk-feat --rootdir ~/datasets/dgl-data/
