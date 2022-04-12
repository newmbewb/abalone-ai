#!/bin/bash

function run(){
    local name=$1
    while :
    do
        pypy ../run/game_generator_single_thread.py $name
    done
}

core_count=$(grep -c ^processor /proc/cpuinfo)
echo "Core Count: ${core_count}"
for i in `seq $core_count`; do
    run $i &
done

wait
