import argparse

import random 
import numpy as np 
import torch 
import json
import tqdm
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
    parser.add_argument('--source', default="HalfCheetah-v2", 
                        help='training interaction/offline-data environment')
    parser.add_argument('--target', default="HalfCheetah1-v2", 
                        help='testing/demo environment')
    # parser.add_argument('--target', default="Hopper1-v2", 
    #                     help='testing/demo environment')
    parser.add_argument('--demo_num', type=int, default=20)
    parser.add_argument('--mode', default="online", help='online or offline-m or offline-mr or offline-me or offline-e')
    parser.add_argument('--demo', default="lfo", help='lfd or lfo')
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--baseline_algo', type=str, default='GAIL')
    parser.add_argument('--alpha', type=int, default=1)
    parser.add_argument('--context_triplet_margin', type=float, default=0.0)
    parser.add_argument('--eval_epsi', type=int, default=100)
    
    # parser.add_argument('--context_size', type=int, default=16)
    parser.add_argument('--window_size', type=int, default=2)
    parser.add_argument('--eval_num', type=int, default=100)
    parser.add_argument('--eval_interval', type=int, default=1)
    parser.add_argument('--algo_steps_per_iter', type=int, default=1000)
    parser.add_argument('--rollout_env_num', type=int, default=10)
    parser.add_argument('--eval_env_num', type=int, default=5)
    parser.add_argument('--eval_src_num', type=int, default=5)
    parser.add_argument('--save_weight', type=bool, default=False)
    parser.add_argument('--use_pretrain', type=bool, default=False)
    parser.add_argument('--eval', type=bool, default=True)
    parser.add_argument('--gen_train_timesteps', type=int, default=None)
    parser.add_argument('--gen_batch_size', type=int, default=2048)
    
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
                "ceil/logs_baseline/{}/{}_{}_{}_{}_{}/Gen_Batch{}_Buffer{}_Steps{}_TotalSteps{}/{}_seed{}_evalNum{}_interval{}_{}", 
                args.baseline_algo, 
                'Cross' if args.source != args.target else 'Single', args.target, args.mode, args.demo, args.demo_num,
                args.gen_batch_size, 1000 if "heetah" in args.source else 200, args.algo_steps_per_iter, args.gen_train_timesteps if args.gen_train_timesteps is not None else args.algo_steps_per_iter * args.rollout_env_num,
                'eval' if args.eval else 'non-eval', args.seed, args.eval_num, args.eval_interval, time_str
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
        replay_buffer_capacity=1000 if "heetah" in args.source else 200,
        algo_n_steps = args.algo_steps_per_iter,
        rollout_env_num=args.rollout_env_num,
        eval_env_num=args.eval_env_num,
        eval_src_num=args.eval_src_num,
        use_pretrain=args.use_pretrain,
        gen_train_timesteps=args.gen_train_timesteps,
        gen_batch_size=args.gen_batch_size
    )
    # learner_rewards_before_training, _ = baseline_trainer.policy_evaluate(args.eval_epsi)
    step = 0
    eval_list=[]
    if args.eval:
        # while step < args.eval_num:
        for i in tqdm.tqdm(range(args.eval_num), desc='eval'):
            baseline_trainer.train_steps(args.algo_steps_per_iter * args.rollout_env_num * args.eval_interval)
            # step += 1
            eval_dict = baseline_trainer.policy_evaluate(args.eval_epsi, non_average=True)
            with baseline_trainer.trainer.logger.accumulate_means("eval"), baseline_trainer.trainer.logger.add_key_prefix(f'eval_step_{i}'):
                assert all(len(v) == args.eval_epsi for v in eval_dict.values()), 'Number of Evaluation eposides are not equivalent'
                for index_value in range(len(list(eval_dict.values())[0])):
                    for key in eval_dict.keys():
                    
                        if 'src' in key:
                            baseline_trainer.trainer.logger.record(f"{args.source if 'heetah' not in args.source else 'hc'}_{args.eval_env_num}env_{key}", eval_dict[key][index_value])
                            
                        elif 'tar' in key:
                            baseline_trainer.trainer.logger.record(f"{args.target if 'heetah' not in args.target else 'hcTarget'}_{args.eval_src_num}env_{key}", eval_dict[key][index_value])
                           
                    baseline_trainer.trainer.logger.dump(step=index_value)
                # for key, value in eval_dict.items():
                #     for index in range(len(value)):
                #         if 'src' in key:
                #             baseline_trainer.trainer.logger.record(f"{args.source}_{args.eval_env_num}env_{key}", value[index])
                #             # baseline_trainer.trainer.logger.record(f"{args.source}_{args.eval_env_num}_{key}_{i}", value[i])
                #             # baseline_trainer.trainer.logger.record(f"{args.source}_env_{args.eval_env_num}/{key}_std", np.std(value))
                #         elif 'tar' in key:
                #             baseline_trainer.trainer.logger.record(f"{args.target}_{args.eval_src_num}env_{key}", value[index])
                #             # baseline_trainer.trainer.logger.record(f"{args.target}_{args.eval_src_num}_{key}_{i}", value[i])
                #             # baseline_trainer.trainer.logger.record(f"{args.target}_env_{args.eval_src_num}/{key}_std", np.std(value))
                #     baseline_trainer.trainer.logger.dump(step=i+1)
            baseline_trainer.trainer.logger.dump(step=i+1)
            eval_list.append(eval_dict)
        eval_path = os.path.join(logger_folder, f'{args.eval_num}_eval_records_interval_{args.eval_interval}.json')
        with open(eval_path, 'w') as f:
            json.dump(eval_list, f, indent=4)
    else:
        pass 


    
    





# hopper 2 0 
