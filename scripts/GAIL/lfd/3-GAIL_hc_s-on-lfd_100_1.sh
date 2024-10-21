#!/bin/bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin

python /storage/songjian/Liu/ceil_code/3train_ceil_debug_baseline.py --source "HalfCheetah-v2" \
    --target "HalfCheetah-v2" --mode "online" --demo "lfd" --baseline_algo "GAIL" --eval_num 100 --eval_interval 1 --eval_epsi 30
