# psize: partition number
# bsize: partitions in mega-batch
# sratio * |V| = static nodes
# bsize/psize + sratio = C

#python3 experiments.py --dataset ogbn-arxiv --method HB --model sage --wt-decay 5e-4  --eval-minibatch 1024  \
#	--part metis --psize 1200 --bsize 20  --sratio 0.083  --recycle 1 --rho 1
#python3 experiments.py --dataset ogbn-arxiv --method HB --model sage --eval-minibatch 1024  \
#	--part metis --psize 1200 --bsize 20  --sratio 0.083  --recycle 1 --rho 1
#python3 experiments.py --dataset ogbn-arxiv --method HB --model sage --eval-minibatch 1024  \
#	--part metis --psize 1200 --bsize 30  --sratio 0.075 --recycle 1 --rho 1
#python3 experiments.py --dataset ogbn-arxiv --method HB --model sage --eval-minibatch 1024  \
#	--part metis --psize 1200 --bsize 40  --sratio 0.067 --recycle 1 --rho 1
#python3 experiments.py --dataset ogbn-arxiv --method HB --model sage --eval-minibatch 1024  \
#	--part metis --psize 1200 --bsize 60  --sratio 0.05 --recycle 1 --rho 1
#python3 experiments.py --dataset ogbn-arxiv --method HB --model sage --eval-minibatch 1024  \
#	--part metis --psize 1200 --bsize 80  --sratio 0.033 --recycle 1 --rho 1
#python3 experiments.py --dataset ogbn-arxiv --method HB --model sage --eval-minibatch 1024  \
#	--part metis --psize 1200 --bsize 90  --sratio 0.025 --recycle 1 --rho 1
#python3 experiments.py --dataset ogbn-arxiv --method HB --model sage --eval-minibatch 1024  \
#	--part metis --psize 1200 --bsize 100  --sratio 0.0167 --recycle 1 --rho 1
#python3 experiments.py --dataset ogbn-arxiv --method HB --model sage --eval-minibatch 1024  \
#	--part metis --psize 1200 --bsize 109  --sratio 0.009167 --recycle 1 --rho 1
#python3 experiments.py --dataset ogbn-arxiv --method HB --model sage --eval-minibatch 1024  \
#	--part metis --psize 1200 --bsize 120  --sratio 0 --recycle 1 --rho 1

#python3 experiments.py --dataset ogbn-papers100M --method HB --model sage --num-hidden 256 --eval-minibatch 1024  \
#	--part metis --psize 18000 --bsize 300 --sratio 0.083 --recycle 1 --rho 1
#python3 experiments.py --dataset ogbn-papers100M --method HB --model sage --num-hidden 256 --eval-minibatch 1024  \
#	--part metis --psize 18000 --bsize 600 --sratio 0.067 --recycle 1 --rho 1
#python3 experiments.py --dataset ogbn-papers100M --method HB --model sage --num-hidden 256 --eval-minibatch 1024  \
#	--part metis --psize 18000 --bsize 900 --sratio 0.05 --recycle 1 --rho 1
#python3 experiments.py --dataset ogbn-papers100M --method HB --model sage --num-hidden 256 --eval-minibatch 1024  \
#	--part metis --psize 18000 --bsize 1800 --sratio 0 --recycle 1 --rho 1

python3 experiments.py --dataset ogbn-papers100M --method HB --model sage --num-hidden 256 --eval-minibatch 1024  \
	--part metis --psize 18000 --bsize 1500 600  --sratio 0.0167 0.067 --recycle 1 --rho 1
#python3 experiments.py --dataset ogbn-papers100M --method HB --model sage --num-hidden 256 --eval-minibatch 1024  \
#	--part metis --psize 18000 --bsize 1500 --sratio 0.0167 --recycle 1 --rho 1
#python3 experiments.py --dataset ogbn-papers100M --method HB --model sage --num-hidden 256 --eval-minibatch 1024  \
#	--part metis --psize 18000 --bsize 1800 --sratio 0 --recycle 1 --rho 1
#python3 experiments.py --dataset ogbn-papers100M --method HB --model sage --num-hidden 256 --eval-minibatch 1024  \
#	--part metis --psize 18000 --bsize 1780 --sratio 0.0011 --recycle 1 --rho 1
#python3 experiments.py --dataset ogbn-papers100M --method HB --model sage --num-hidden 256 --eval-minibatch 1024  \
#	--part metis --psize 18000 --bsize 1636 --sratio 0.0091 --recycle 1 --rho 1
#python3 experiments.py --dataset ogbn-papers100M --method HB --model sage --num-hidden 256 --eval-minibatch 1024  \
#	--part metis --psize 18000 --bsize 1200 --sratio 0.0333 --recycle 1 --rho 1
#python3 experiments.py --dataset ogbn-papers100M --method HB --model sage --num-hidden 256 --eval-minibatch 1024  \
#	--part metis --psize 18000 --bsize 900 --sratio 0.05 --recycle 1 --rho 1
#python3 experiments.py --dataset ogbn-papers100M --method HB --model sage --num-hidden 256 --eval-minibatch 1024  \
#	--part metis --psize 18000 --bsize 100 --sratio 0.0944 --recycle 1 --rho 1
#python3 experiments.py --dataset ogbn-papers100M --method HB --model sage --num-hidden 256 --eval-minibatch 1024  \
#	--part metis --psize 18000 --bsize 1700 --sratio 0.0056 --recycle 1 --rho 1
#python3 experiments.py --dataset ogbn-papers100M --method HB --model sage --num-hidden 256 --eval-minibatch 1024  \
#	--part metis --psize 18000 --bsize 300 --sratio 0.0833 --recycle 1 --rho 1
