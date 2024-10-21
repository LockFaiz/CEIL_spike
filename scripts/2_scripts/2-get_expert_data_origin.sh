#!/bin/bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin
# envs=("Hopper-v2" "HalfCheetah-v2" "Walker2d-v2" "Ant-v2")
envs=("Ant-v2")
for env in "${envs[@]}"
do
    python /storage/songjian/Liu/ceil_code/2run_gen_expert_data.py --env "$env"
done