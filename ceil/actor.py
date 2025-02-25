from stable_baselines3.common.torch_layers import (
    create_mlp,
)

import torch as th
from torch import nn
import torch.nn.functional as F

import math
from torch.distributions import Normal
from spikingjelly.activation_based import layer, neuron, functional, surrogate

def normal_entropy(std):
    var = std.pow(2)
    entropy = 0.5 + 0.5 * th.log(2 * var * math.pi)
    return entropy.sum(1, keepdim=True)

def normal_log_density(x, mean, log_std, std):
    var = std.pow(2)
    log_density = -(x - mean).pow(2) / (2 * var) - 0.5 * math.log(2 * math.pi) - log_std
    return log_density.sum(1, keepdim=True)


def weights_init_(m):
    if isinstance(m, nn.Linear):
        th.nn.init.xavier_uniform_(m.weight, gain=0.1)
        th.nn.init.constant_(m.bias, 0.)


class ContextualActor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        context_size: int,
        action_dim: int,
        output_max: float,
        emb_size: int = 32,
        hidden_size: int = 512,
        log_std: float = -6,
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.context_size = context_size
        self.action_dim = action_dim
        self.output_max = output_max

        self.enc_ts = nn.Embedding(1001, emb_size)
        self.enc_obs = nn.Linear(self.obs_dim, emb_size)
        self.enc_con = nn.Linear(self.context_size, emb_size)

        # self.fc1 = nn.Linear(emb_size * 2, hidden_size)
        self.fc1 = nn.Linear(self.obs_dim + self.context_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.action_mean = nn.Linear(hidden_size, self.action_dim)

        self.noise = th.Tensor(self.action_dim)
        
        self.apply(weights_init_)
        

    def forward(self, obs: th.Tensor, context: th.Tensor,) -> th.Tensor:
        if len(context.shape) == 1:
            context = context.unsqueeze(0).repeat(obs.shape[0], 1)
        
        # ts = self.enc_ts(ts) 
        # obs = self.enc_obs(obs)
        # context = self.enc_con(context)
        x = th.concat([obs, context,], -1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        action_mean = self.action_mean(x)
        action_mean = self.output_max * th.tanh(action_mean)

        noise = self.noise.normal_(0., std=0.001).to(obs.device)
        noise = noise.clamp(-0.1, 0.1)
        action = action_mean + noise
        
        return action_mean, action 
    
    def select_action(self, x, c):
        action_mean, _ = self.forward(x, c)
        return action_mean
    
class ContextualActorSpike(ContextualActor):
    def __init__(self, 
                obs_dim: int,
                context_size: int,
                action_dim: int,
                output_max: float,
                T: int = 4,
                use_cupy = True,
                neuron_name: str = "LIFNode",
                emb_size: int = 32,
                hidden_size: int = 512,
                log_std: float = -6,):
        self.T = T
        self.neuron = neuron_name

        try:
            self.node = getattr(neuron, self.neuron)
        except Exception as e:
            print(e)
        else:
            print(f'Actor network will use SpikingJelly\'s {self.neuron} as Activation function')
        super().__init__(obs_dim,
                       context_size, 
                       action_dim, 
                       output_max,
                       emb_size,
                       hidden_size,
                       log_std)
        self.ActorNet = nn.Sequential(
            layer.Linear(self.obs_dim+self.context_size, hidden_size),
            self.node(surrogate_function = surrogate.ATan()),
            layer.Linear(hidden_size, hidden_size),
            self.node(surrogate_function = surrogate.ATan()),
            layer.Linear(hidden_size, hidden_size),
            self.node(surrogate_function = surrogate.ATan()),
        )

        # self.action_mean = layer.Linear(hidden_size, self.action_dim)
        if use_cupy:
            functional.set_backend(self, backend='cupy')
        if self.T > 1:
            functional.set_step_mode(self, step_mode='m')

        # self.apply(weights_init_)
    def forward(self, obs: th.Tensor, context: th.Tensor,) -> th.Tensor:
        if len(context.shape) == 1:
            context = context.unsqueeze(0).repeat(obs.shape[0], 1)
        x = th.concat([obs, context,], -1)
        if self.T > 1:
            x_seq = x.unsqueeze(0).repeat(self.T, *[1 for i in range(len(x.shape))])
            x_seq = self.ActorNet(x_seq).mean(0)
        elif self.T == 1:
            x_seq = self.ActorNet(x)

        action_mean = self.action_mean(x_seq)
        action_mean = self.output_max * th.tanh(action_mean)
        noise = self.noise.normal_(0., std=0.001).to(obs.device)
        noise = noise.clamp(-0.1, 0.1)
        action = action_mean + noise
        functional.reset_net(self.ActorNet)
        return action_mean, action 