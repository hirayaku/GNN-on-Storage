
dataset=${dataset:-ogbn-papers100M}
part=${part:-metis}
minibatch=${minibatch:-1024}
psize=${psize:-1024}
hsizes=(512)
bsizes=(512)
pratios=(0.01)
lr=${lr:-0.001}
lr_decay=${lr_decay:-0.9999}
recycle=${recycle:-1}
rho=${rho:-1}
comment=${comment:-"new run"}

runs=${#hsizes[@]}
for (( i=0; i<$runs; i++)); do
    hsize=${hsizes[$i]}
    bsize=${bsizes[$i]}
    pratio=${pratios[$i]}
    echo "$dataset, $part-$psize, ($hsize, $bsize), $(bc -l <<< "$pratio*100")%"
    PSIZE=$psize HSIZE=$hsize BSIZE=$bsize BSIZE2=$minibatch PRATIO=$pratio LR=$lr LRD=$lr_decay \
    PART=$part REC=$recycle RHO=$rho COMMENT="$comment" \
    sbatch -J $dataset-$part-$psize/$hsize/$bsize/$minibatch/$pratio \
        --export=ALL sbatch/$dataset.sbatch
done

