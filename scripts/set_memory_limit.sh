#!/bin/bash

if [[ "$1" =~ "-h" ]]; then
    printf "Usage:\n\t$0 [cap] [user] [group]\n"
    exit 0
fi

MEM_CAP=$1
USERNAME=$2
GROUPNAME=$3
[[ -z $USERNAME ]] && USERNAME=$USER
[[ -z $GROUPNAME ]] && GROUPNAME=$USERNAME

cgcreate -a $USERNAME:$GROUPNAME -t $USERNAME:$GROUPNAME -g memory:GNN
echo $MEM_CAP > /sys/fs/cgroup/memory/GNN/memory.limit_in_bytes
echo 0 > /sys/fs/cgroup/memory/GNN/memory.swappiness

