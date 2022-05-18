# in-memory execution
#python graphsage.py --gpu -1 --dataset ogbn-products

# in-disk execution
python graphsage.py --gpu -1 --dataset ogbn-products --rootdir ~/datasets/dgl/ --data-cpu --disk-feat
