from absim.gym_wrapper import AbstractSim, wrapper_dict

import numpy
from stable_baselines import DQN, PPO2, A2C, ACKTR
from stable_baselines.common.cmd_util import make_vec_env
from stable_baselines.common.callbacks import BaseCallback

import pickle
import argparse
import json
from datetime import datetime
import os

parser = argparse.ArgumentParser(description = 'Train scavenger hunt in graph world environment')
parser.add_argument('--config', dest = 'config_path', type = str, default = './config/default_sb.json', help = 'path to the configuration file')
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

if config['wrapper']:
    env = wrapper_dict[config['wrapper']](AbstractSim(), config["wrapper_args"], world_file = config['world_file']+'train.dat')
    train_envs = make_vec_env(lambda: wrapper_dict[config['wrapper']](AbstractSim(), config["wrapper_args"], world_file = config['world_file']+'train.dat'), n_envs=1)
else:
    env = AbstractSim(world_file = config['world_file'])
    train_envs = make_vec_env(lambda: AbstractSim(world_file = config['world_file']), n_envs=1)

if config['algorithm'] == 'PPO2':
    model = PPO2(config['policy_network'], train_envs,
                    learning_rate = config['learning_rate'],
                    gamma = config['gamma'], policy_kwargs = config['policy_kwargs'],
                    verbose=1, tensorboard_log = save_path)

model.learn(config['total_steps'])
model.save(os.path.join(save_path, 'model'))
