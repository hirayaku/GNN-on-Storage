#!/bin/bash

if [[ "$1" =~ "-h" || $# == 0 ]]; then
    printf "Usage:\n\t$0 [cap] [user] [group]\n"
    exit 0
fi

MEM_CAP=$1
user=$2
group=$3
[[ -z $user ]] && user=$USER
[[ -z $group ]] && group=$user

cgroup=gnn$MEM_CAP
set -x

cgcreate -a $user:$group -t $user:$group -g memory:$cgroup
echo $MEM_CAP > /sys/fs/cgroup/memory/$cgroup/memory.limit_in_bytes
echo 0 > /sys/fs/cgroup/memory/$cgroup/memory.swappiness

set +x
echo "Memory limited to $MEM_CAP for memory:$cgroup"
