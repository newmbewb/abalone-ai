#!/bin/bash

function run(){
    local name=$1
    pypy ../run/dataset_generator_single_thread.py $name
}

core_count=$(grep -c ^processor /proc/cpuinfo)
echo "Core Count: ${core_count}"
for i in `seq $core_count`; do
    run $i &
done

wait
