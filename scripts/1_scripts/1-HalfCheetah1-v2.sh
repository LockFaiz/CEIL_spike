#!/bin/bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin
python /storage/songjian/Liu/ceil_code/1run_learn_expert_policy.py --env HalfCheetah1-v2
