set -x

sudo bash scripts/drop_cache.sh
cgexec -g memory:gnn128G python3 trainer_ns.py --dataset ogbn-papers100M --root /mnt/data/datasets/gnnos --mmap --model sage --num-hidden 256 --fanout 15,10,5 --n-epochs 5 --num-workers 16 | tee mmap/papers-sage.log

# sudo bash scripts/drop_cache.sh
# cgexec -g memory:gnn128G python3 trainer_ns.py --dataset ogbn-papers100M --root /mnt/data/datasets/gnnos --mmap --model gat --mlp --num-hidden 1024 --fanout 15,10,5 --n-epochs 5 --num-workers 16 | tee mmap/papers-gat.log

# sudo bash scripts/drop_cache.sh
# cgexec -g memory:gnn128G python3 trainer_ns.py --dataset ogbn-papers100M --root /mnt/data/datasets/gnnos --mmap --model gin --num-hidden 512 --fanout 20,20,20 --n-epochs 5 --num-workers 16 | tee mmap/papers-gin.log

