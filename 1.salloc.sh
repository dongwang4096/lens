#!/bin/bash
PARTITION=${PARTITION:-"batch"}
# PARTITION=${PARTITION:-"36x2-a01r"}
echo "PARTITION: ${PARTITION}"
export WORLD_SIZE=1
salloc -A coreai_devtech_all -J coreai_devtech_all-test:test  --gpus-per-node=4 -p ${PARTITION} -N ${WORLD_SIZE} --segment=${WORLD_SIZE} --exclusive --mem 0 --time 4:00:00
