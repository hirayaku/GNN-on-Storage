
export PYTHONPATH="${PYTHONPATH}:/home/cxh/proj/GNNoS"

#python trainer_mmap.py --lr 0.001 --num-hidden 256 --n-epochs 5 --dropout 0.5 --fanout 15,10,5 --dataset ogbn-products --mmap --model sage
python trainer_mmap.py --lr 0.001 --num-hidden 256 --n-epochs 6 --dropout 0.5 --fanout 15,10,5 --dataset ogbn-papers100M --mmap --model sage &> ogbn-papers100M-128G-2.log &
#python trainer_mmap.py --lr 0.001 --num-hidden 1024 --n-epochs 6 --dropout 0.5 --n-layers 2 --fanout 25,15 --test-fanout 25,15 --dataset mag240m --mmap --model sage &> mag240m-144G.log &
