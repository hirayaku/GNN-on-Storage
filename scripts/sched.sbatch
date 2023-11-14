#!/bin/bash
#SBATCH -p tig
#SBATCH -q tig-main
#SBATCH -x tig-slurm-4
#SBATCH -c 32
#SBATCH -G 1
#SBATCH --mem=0
##SBATCH --exclusive
#SBATCH -t 24:00:00

date
source $HOME/.profile
conda activate ml

set -ex
cd $HOME/proj/GNNoS-current
if [[ $# -gt 0 ]]; then
    python3 $*
else
    echo "no script given"
fi

