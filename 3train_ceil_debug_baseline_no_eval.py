import argparse

import random 
import numpy as np 
import torch 
import json

from imitation.rewards.reward_nets import BasicRewardNet

# from ceil.actor import ContextualActor, ContextualActorSpike
# from ceil.encoder import TrajEncoder
# from ceil.encoder_ import TrajEncoder
# from ceil.encoder_mlp import TrajEncoder, TrajEncoderSpike
from ceil.ceil_baseline import BaselineTrainer

import pickle
import os.path as osp, time, atexit, os

import datetime

def setup(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dtype = torch.float32
    torch.set_default_dtype(dtype)

def get_args():
    parser = argparse.ArgumentParser(description='CEIL arguments')
    # ["Hopper-v2", "HalfCheetah-v2", "Walker2d-v2", "Ant-v2"]
    parser.add_argument('--source', default="Ant-v2", 
                        help='training interaction/offline-data environment')
    parser.add_argument('--target', default="Ant-v2", 
                        help='testing/demo environment')
    # parser.add_argument('--target', default="Hopper1-v2", 
    #                     help='testing/demo environment')
    parser.add_argument('--demo_num', type=int, default=20)
    parser.add_argument('--mode', default="online", help='online or offline-m or offline-mr or offline-me or offline-e')
    parser.add_argument('--demo', default="lfd", help='lfd or lfo')
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--baseline_algo', type=str, default='AIRL')
    parser.add_argument('--alpha', type=int, default=1)
    parser.add_argument('--context_triplet_margin', type=float, default=0.0)
    parser.add_argument('--eval_epsi', type=int, default=100)
    
    # parser.add_argument('--context_size', type=int, default=16)
    parser.add_argument('--window_size', type=int, default=2)
    parser.add_argument('--eval_num', type=int, default=100)
    parser.add_argument('--eval_interval', type=int, default=1)
    parser.add_argument('--algo_steps_per_iter', type=int, default=2048)
    parser.add_argument('--rollout_env_num', type=int, default=10)
    parser.add_argument('--eval_env_num', type=int, default=5)
    parser.add_argument('--eval_src_num', type=int, default=5)
    parser.add_argument('--save_weight', type=bool, default=False)
    parser.add_argument('--use_pretrain', type=bool, default=False)
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    setup(args.seed)
    
    assert args.source in ["Hopper-v2", "HalfCheetah-v2", "Walker2d-v2", "Ant-v2"]
    assert args.target in ["Hopper-v2", "HalfCheetah-v2", "Walker2d-v2", "Ant-v2", 
                 "Hopper1-v2", "HalfCheetah1-v2", "Walker2d1-v2", "Ant1-v2", ]
    assert args.mode in ["online", "offline-m", "offline-mr", "offline-me", "offline-e"]
    assert args.demo in ["lfd", "lfo"]
    abs_dir = os.path.dirname(os.path.abspath(__file__))
    with open(osp.join(abs_dir, "_data/demo/", args.target+"_"+str(20)+".pkl"), "rb") as savepkl: 
        demonstrations = pickle.load(savepkl) # TrajectoryWithRew 

    time_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    logger_folder=str.format(
                "ceil/logs_baseline/{}/{}_{}_{}_{}_{}/{}_seed{}_eval{}_interval{}_{}", 
                args.baseline_algo, 
                args.source, args.target, args.mode, args.demo, args.demo_num,
                'download' if args.use_pretrain else 'Local', args.seed, args.eval_num, args.eval_interval, time_str
            )
    logger_folder = os.path.join(abs_dir,logger_folder)
    if args.save_weight:
        pth_folder = f'pth_{args.baseline_algo}/{args.source}_{args.target}_{args.mode}_{args.demo_num}_{args.demo}/{time_str}_window{args.window_size}_seed{args.seed}'
        pth_folder = os.path.join(abs_dir, pth_folder)
        if not os.path.exists(pth_folder):
            os.makedirs(pth_folder)
    else:
        pth_folder = None
    baseline_trainer = BaselineTrainer(
        algo = args.baseline_algo,
        source_env_name=args.source,
        target_env_name=args.target,
        mode=args.mode,
        demo=args.demo,
        demonstrations=demonstrations[:args.demo_num],
        seed=args.seed,
        device="cuda",
        window_size=args.window_size,
        # disc_net=BasicRewardNet,
        c_logger_folder=logger_folder,
        allow_variable_horizon=True,
        replay_buffer_capacity=200,
        algo_n_steps = args.algo_steps_per_iter,
        rollout_env_num=args.rollout_env_num,
        eval_env_num=args.eval_env_num,
        eval_src_num=args.eval_src_num,
        use_pretrain=args.use_pretrain
    )
    # learner_rewards_before_training, _ = baseline_trainer.policy_evaluate(args.eval_epsi)
    step = 0
    eval_list=[]
    while step < args.eval_num:
        baseline_trainer.train_steps(args.algo_steps_per_iter * args.rollout_env_num * args.eval_interval)
        step += 1
        eval_dict = baseline_trainer.policy_evaluate(args.eval_epsi, non_average=True)
        with baseline_trainer.trainer.logger.accumulate_means("eval"), baseline_trainer.trainer.logger.add_key_prefix(f'eval_step_{step}'):
            for key, value in eval_dict.items():
                for i in range(len(value)):
                    if 'src' in key:
                        baseline_trainer.trainer.logger.record(f"{args.source}_{args.eval_env_num}_{key}", value[i])
                        # baseline_trainer.trainer.logger.record(f"{args.source}_{args.eval_env_num}_{key}_{i}", value[i])
                        # baseline_trainer.trainer.logger.record(f"{args.source}_env_{args.eval_env_num}/{key}_std", np.std(value))
                    elif 'tar' in key:
                        baseline_trainer.trainer.logger.record(f"{args.target}_{args.eval_src_num}_{key}", value[i])
                        # baseline_trainer.trainer.logger.record(f"{args.target}_{args.eval_src_num}_{key}_{i}", value[i])
                        # baseline_trainer.trainer.logger.record(f"{args.target}_env_{args.eval_src_num}/{key}_std", np.std(value))
                    baseline_trainer.trainer.logger.dump(step=i)
        baseline_trainer.trainer.logger.dump(step)
        eval_list.append(eval_dict)
    eval_path = os.path.join(logger_folder, f'{args.eval_num}_eval_records_interval_{args.eval_interval}.json')
    with open(eval_path, 'w') as f:
        json.dump(eval_list, f, indent=4)



    
    





# hopper 2 0 



 
