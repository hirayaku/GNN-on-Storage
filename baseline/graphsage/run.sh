# in-memory execution
#python graphsage.py --gpu -1 --dataset ogbn-products --num-epochs 2

# in-disk execution
#python graphsage.py --gpu -1 --dataset ogbn-products --rootdir ~/datasets/dgl-data/ --data-cpu --disk-feat --num-epochs 2

#python multi_gpu_node_classification.py --dataset ogbn-products --num-epochs=2 --gpu 0

#python multi_gpu_node_classification.py --dataset ogbn-products --num-epochs=20 --gpu 0  --eval-every 5

python multi_gpu_node_classification.py --dataset ogbn-products --num-epochs=20 --gpu 0 --disk-feat --rootdir ~/datasets/dgl-data/
