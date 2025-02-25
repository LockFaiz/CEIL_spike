from gym.envs.mujoco import HalfCheetahEnv
import gym 

import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.networks import ConcatMlp
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
import os 
import argparse

def experiment(variant, env_name):
    expl_env = NormalizedBoxEnv(gym.make(env_name))
    eval_env = NormalizedBoxEnv(gym.make(env_name))
    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size

    M = variant['layer_size']
    qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M],
    )
    eval_policy = MakeDeterministic(policy)
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
        policy,
    )
    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )
    trainer = SACTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['trainer_kwargs']
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()




if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algorithm="SAC",
        version="normal",
        layer_size=256,
        replay_buffer_size=int(1E6),
        algorithm_kwargs=dict(
            num_epochs=3000,
            num_eval_steps_per_epoch=5000,
            num_trains_per_train_loop=1000,
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=1000,
            max_path_length=1000,
            batch_size=256,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=1,
            use_automatic_entropy_tuning=True,
        ),
    )
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default=None, help="choose a single env from Hopper1-v2,HalfCheetah1-v2,Walker2d1-v2,Ant1-v2")
    args = parser.parse_args()
    # assert args.env in ["Hopper1-v2","HalfCheetah1-v2","Walker2d1-v2","Ant1-v2"], 'You give a new env.'
    # env_name="Hopper1-v2" # 3234
    # env_name="HalfCheetah1-v2" # 12135
    # env_name="Walker2d1-v2" # 4592
    # env_name="Ant1-v2"

    # env_name="HalfCheetah-v2" 
    env_name = args.env
    save_dir = os.path.dirname(os.path.abspath(__file__))
    # setup_logger('sac_'+env_name, 
    #              variant=variant, 
    #              log_dir="./_data/experts-new/"+env_name+"/")
    setup_logger('sac_'+env_name, 
                 variant=variant, 
                #  log_dir=os.path.join(save_dir, "_data/experts-new/", env_name,'retry'))
                 log_dir=os.path.join(save_dir, "_data/experts-new/", env_name,))
    ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    experiment(variant, env_name)
    
# CUDA_VISIBLE_DEVICES=