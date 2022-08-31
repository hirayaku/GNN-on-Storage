
dataset=${dataset:-ogbn-papers100M}
part=${part:-metis}
minibatch=${minibatch:-1024}
psize=${psize:-1024}
hsizes=(1)
bsizes=(1)
pratios=(0)
lr=${lr:-0.001}
lr_decay=${lr_decay:-0.9999}
comment=${comment:-"new run"}

runs=${#hsizes[@]}
for (( i=0; i<$runs; i++)); do
    hsize=${hsizes[$i]}
    bsize=${bsizes[$i]}
    pratio=${pratios[$i]}
    echo "$dataset, $part-$psize, ($hsize, $bsize), $pratio%"
    PSIZE=$psize HSIZE=$hsize BSIZE=$bsize BSIZE2=$minibatch PRATIO=$pratio LR=$lr LRD=$lr_decay \
    PART="$part" COMMENT="$comment" sbatch -J $dataset-sage-$minibatch \
        --export=ALL sbatch/$dataset-sg.sbatch
done

