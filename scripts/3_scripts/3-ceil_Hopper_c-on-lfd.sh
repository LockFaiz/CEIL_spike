#!/bin/bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin

python /storage/songjian/Liu/ceil_code/3train_ceil.py --source "Hopper-v2" \
    --target "Hopper1-v2" --mode "online" --demo "lfd"
