# in-memory execution
#python graphsage.py --gpu -1 --dataset ogbn-products --num-epochs 2

# in-disk execution
python graphsage.py --gpu -1 --dataset ogbn-products --rootdir ~/datasets/dgl/ --data-cpu --disk-feat --num-epochs 2
