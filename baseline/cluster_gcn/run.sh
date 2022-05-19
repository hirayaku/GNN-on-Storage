# in-memory execution on CPU
#python cluster_gcn.py --dataset ogbn-products --num-epochs 2 --gpu -1

# in-disk execution on CPU
#python cluster_gcn.py --dataset ogbn-products --num-epochs 2 --gpu -1 --disk-feat --rootdir ~/datasets/dgl-data/ --data-cpu

# in-memory execution on GPU
#python cluster_gcn.py --dataset ogbn-products --num-epochs 2 --gpu 0

# in-disk execution on GPU
python cluster_gcn.py --dataset ogbn-products --num-epochs 2 --gpu 0 --disk-feat --rootdir ~/datasets/dgl-data/  --data-cpu
