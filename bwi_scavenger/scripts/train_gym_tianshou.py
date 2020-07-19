from absim.gym_wrapper import AbstractSim, wrapper_dict

import numpy
import tianshou as ts
from tianshou_policies import *
import torch, numpy as np
from torch import nn
from torch.utils.tensorboard import SummaryWriter

import pickle
import argparse
import json
from datetime import datetime
import os

parser = argparse.ArgumentParser(description = 'Train scavenger hunt in graph world environment')
parser.add_argument('--config', dest = 'config_path', type = str, default = './config/default.json', help = 'path to the configuration file')
parser.add_argument('--save', dest = 'save_path', type = str, default = './results/', help = 'path to the saving folder')

args = parser.parse_args()
config_path = args.config_path
save_path = args.save_path

if not config_path.startswith('./'):
    config_path = './config/' + config_path

with open(config_path, 'rb') as f:
    config = json.load(f)

now = datetime.now()
dt_string = now.strftime("%Y_%m_%d_%H_%M")
save_path = os.path.join(save_path, config['section'] + "_" + dt_string)
if not os.path.exists(save_path):
    os.mkdir(save_path)

with open(os.path.join(save_path, 'config.json'), 'w') as fp:
    json.dump(config, fp)

writer = SummaryWriter(save_path)

if config['wrapper']:
    env = wrapper_dict[config['wrapper']](AbstractSim(), config["wrapper_args"], world_file = config['world_file']+'train.dat')
    train_envs = ts.env.VectorEnv([lambda:wrapper_dict[config['wrapper']](AbstractSim(), config["wrapper_args"], world_file = config['world_file']+'train.dat') for _ in range(1)])
    test_envs = ts.env.VectorEnv([lambda: wrapper_dict[config['wrapper']](AbstractSim(), config["wrapper_args"], world_file = config['world_file']+'test.dat') for _ in range(10)])
else:
    env = AbstractSim(world_file = config['world_file'])
    train_envs = ts.env.VectorEnv([lambda: AbstractSim(world_file = config['world_file']) for _ in range(1)])
    test_envs = ts.env.VectorEnv([lambda: AbstractSim(world_file = config['world_file']) for _ in range(10)])

state_shape = env.observation_space.shape or env.observation_space.n
action_shape = env.action_space.shape or env.action_space.n

policy_net = policies_dic[config['policy_network']](state_shape, action_shape)

if config['optim'] == 'Adam':
    optim = torch.optim.Adam(policy_net.parameters(), lr=config['learning_rate'])

if config['algorithm'] == 'DQNPolicy':
    policy = ts.policy.DQNPolicy(policy_net, optim,
                                discount_factor=config['gamma'], estimation_step=3,
                                use_target_network=True, target_update_freq=512)
elif config['algorithm'] == 'PGPolicy':
    policy = ts.policy.PGPolicy(policy_net, optim,
                                discount_factor=config['gamma'])
elif config['algorithm'] == 'PPOPolicy':
    policy = ts.policy.PPOPolicy(policy_net, optim, discount_factor=config['gamma'])

if config['base_model']:
    state_dict = torch.load('./results/' + config['base_model'] + '/dqn.pth')
    policy.load_state_dict(state_dict)

train_collector = ts.data.Collector(policy, train_envs, ts.data.ReplayBuffer(size=config['buffer_size']))
test_collector = ts.data.Collector(policy, test_envs)

if config['algorithm'] == 'DQNPolicy':
    result = ts.trainer.offpolicy_trainer(
        policy, train_collector, test_collector,
        max_epoch=config['max_epoch'],
        step_per_epoch=config['step_per_epoch'],
        collect_per_step=config['collect_per_step'],
        episode_per_test=200, batch_size=64,
        train_fn=lambda e: policy.set_eps(max(0.02, 1-0.1*e)),
        test_fn=lambda e: policy.set_eps(0.0),
        save_fn=lambda e: torch.save(policy.state_dict(), save_path + '/dqn.pth'),
        # stop_fn=lambda x: x >= env.spec.reward_threshold,
        writer=writer)
else:
    result = ts.trainer.onpolicy_trainer(
        policy, train_collector, test_collector,
        max_epoch=config['max_epoch'],
        step_per_epoch=config['step_per_epoch'],
        collect_per_step=config['collect_per_step'],
        repeat_per_collect = 1, 
        episode_per_test=200, batch_size=64,
        save_fn=lambda e: torch.save(policy.state_dict(), save_path + '/model.pth'),
        writer = writer)
print(f'Finished training! Use {result["duration"]}')
