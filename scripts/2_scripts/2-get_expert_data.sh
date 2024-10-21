#!/bin/bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin
envs=("Hopper1-v2" "HalfCheetah1-v2" "Walker2d1-v2" "Ant1-v2")
for env in "${envs[@]}"
do
    python /storage/songjian/Liu/ceil_code/2run_gen_expert_data.py --env "$env"
done