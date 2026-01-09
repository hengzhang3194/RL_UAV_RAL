import sys
import os
sys.path.append('../')
# sys.path.insert(0, 'e:\\Zhangheng\\DATT_v2\\DATT_new')

import numpy as np
import time
import csv

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Normal

import gymnasium as gym

from envs.sac_agent import SacAgent, ReplayBuffer
from collections import defaultdict
from envs.env_flare import DroneEnv as DroneEnv_flare
from envs.env_model import DroneEnv as DroneEnv_model
from envs.desired_trajectory import Desired_trajectory
from envs.controller_att import Controller_Attitude
from envs.sac_agent import ContinuousPolicyNetwork

import pandas as pd
import pdb


# drone = DroneEnv_flare(localhost=25556)
drone = DroneEnv_model()
desired_trajectory = Desired_trajectory(trajectory_flag='horizon_eight')
controller_att = Controller_Attitude()

log = defaultdict(list)  # 用于存储信息的字典

# 全局调整 NumPy 数组的打印格式
np.set_printoptions(precision=3, suppress=True, floatmode='fixed', linewidth=150)  # useful for printing


# 位置环 RL，姿态环采取非线性反馈控制
obs_dims = 12
act_dims = 3
hidden_size=[256, 256]

pi_model_path = 'tensorboard/Drone_model/SAC/20251226_144738/ckpts/latest/pi.pth'

assert os.path.exists(pi_model_path), f"Path '{pi_model_path}' of policy model DOESN'T exist."

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pi_net = ContinuousPolicyNetwork(obs_dims, act_dims, hidden_size).to(device)
pi_net.load_state_dict(torch.load(pi_model_path))
pi_net.eval()  # 切换到eval模式，让NN的预测更稳定。



obs, _ = drone.reset()
obs_flag = False
deterministic=True
action = np.zeros(3)

start_timestamp = time.perf_counter()
# pdb.set_trace()
while (drone.time < drone.duration):
    print(f"Now is {drone.time}.")
    
    if obs_flag:
        obs_tensor = torch.FloatTensor(obs).to(device)
        mean, log_std = pi_net(obs_tensor)
        std = log_std.exp()
        

        
        if deterministic:
            action = torch.tanh(mean)
        else:
            z = Normal(0, 1).sample(mean.shape).to(device)
            action = torch.tanh(mean + std * z)
        action = action.detach().cpu().numpy()

    obs_flag = False
        

    next_obs = obs


    # 仿真姿态环，确保obs是位置环的频率。
    # while obs_flag==False:
    # 获取当前时刻的期望轨迹的信息
    state_des = desired_trajectory.get_desired_trajectory(drone.time)

    att = next_obs[6:9] + state_des.att
    ang = next_obs[9:] + state_des.ang
    scale = np.array([6, 6, 13])    # 13 = 1.32*9.8
    act_att = scale * np.array([action[0], action[1], action[2]])
    act_att = controller_att.NFC_att(act_att, att, ang, state_des)
    next_obs, reward, terminated, truncated, info = drone.step(act_att, state_des)
    obs_flag = info["obs_flag"]
    print(f"Obs is {next_obs[:3]}.")

            
    obs = next_obs

    if terminated or truncated:
        break
        


    