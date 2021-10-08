#!/usr/bin/bash

cgcreate -a tianhaoh:tianhaoh -t tianhaoh:tianhaoh -g memory:GNN
echo 64G > /sys/fs/cgroup/memory/GNN/memory.limit_in_bytes
echo 0 > /sys/fs/cgroup/memory/GNN/memory.swappiness

