# python3 experiments.py --dataset mag240m-c --num-hidden 1024 --layers 2 --fanout 25,15 --test-fanout 25,15 --psize 16384 --bsize 1024 \
#     --minibatch 1024 --eval-minibatch 1024 --method NS HB HB HB --sratio 0 0 0.01 0.01 --recycle 1 1 1 2 --lr 0.001 --epochs 25 --eval-every 1 --log-every 200 \
#     --model gin --num-workers 4 --mem 768G --gpu 0 1 0 1 --runs 4 --jobs 2 --comment mag240m-c-gin
#
python3 experiments.py --dataset mag240m-c --method NS HB HB HB --sratio 0 0.01 0 0.01 --recycle 1 1 1 2 --jobs 2 --extra 'mlp' --gpu 0 1 0 1 \
--num-hidden 1024 --minibatch 1024 --eval-minibatch 1024 --layers 2 --fanout 25,15 --test-fanout 25,15 --epochs 25 --eval-every 1 --log-every 200  \
--model sage --num-workers 4 --runs 4 --mem 768G --jobs 2 --comment mag240m-c
