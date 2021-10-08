#!/bin/bash

echo "# $(date)" > results/io.trace
# (iostat -y -x /dev/sda2 /dev/md0 -t 1 | awk '/md0/ {print strftime("%H:%M:%S", systime()), $0 } /sda/ {print strftime("%H:%M:%S", systime()), $0}' >> io.trace) &
(iostat -y -x /dev/sda2 /dev/md0 -t 1 | awk '/md0/ {print $0 } /sda/ {print $0}' >> results/io.trace) &

echo "# $(date)" > results/cpu.trace
(iostat -y -c -t 1 | awk '{if(p==1){print $1, $3, $4, $6}} {if($1 ~ /cpu/) {p=1} else {p=0}}' >> results/cpu.trace) &

echo "# $(date)" > results/mem.trace
# (dstat --mem-adv | awk '{if($1 ~ /[0-9]/) {print}}' >> mem.trace)
(dstat --mem-adv -s >> results/mem.trace)

