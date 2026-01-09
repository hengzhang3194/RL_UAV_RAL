import sys
import os
sys.path.append('../')
# sys.path.insert(0, 'e:\\Zhangheng\\DATT_v2\\DATT_new')

import numpy as np
import time
import csv

import torch
from torch.utils.tensorboard import SummaryWriter

import gymnasium as gym

from envs.sac_agent import SacAgent, ReplayBuffer
from collections import defaultdict
from envs.env_flare import DroneEnv as DroneEnv_flare
from envs.env_model import DroneEnv as DroneEnv_model
from envs.env_nonlinear_model import DroneEnv as DroneEnv_nomodel
from envs.desired_trajectory import Desired_trajectory
from envs.controller_att import Controller_Attitude

import pandas as pd
import pdb


# drone = DroneEnv_flare(localhost=25556)
drone = DroneEnv_nomodel()
desired_trajectory = Desired_trajectory(trajectory_flag='horizon_eight')
controller_att = Controller_Attitude()

log = defaultdict(list)  # 用于存储信息的字典

# 全局调整 NumPy 数组的打印格式
np.set_printoptions(precision=3, suppress=True, floatmode='fixed', linewidth=150)  # useful for printing


hidden_dims = [256, 256]
gamma = 0.99
tau = 0.005
q_lr = 3e-4
pi_lr = 3e-4
a_lr = 3e-4

max_steps = 6e6
buffer_size = 1e6
batch_size = 256
model_save_interval = 10
average_range = 50
return_threshold = np.inf


# buffer
buffer = ReplayBuffer(buffer_size)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
agent = SacAgent(drone.observation_space.shape[0], drone.action_space.shape[0], hidden_dims, gamma, tau, q_lr, pi_lr, a_lr, device=device)
# agent.load_model('./tensorboard/Drone_model_Env_v0/SAC/abf467cf/ckpts/latest')    # 从某一个ckpt恢复训练，而不是从0开始


# logger
timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())    # 设置该格式的时间戳，
logger = SummaryWriter(log_dir=f"./tensorboard/drone_model/{agent.name}/{timestamp}")    # 在该路径下记录训练过程中的数据

# training
steps, episodes, returns = 0, 0, []
while steps <= max_steps:
    episode_reward = 0
    episode_len = 0

    obs, _ = drone.reset()
    obs_flag = False
    mass = 1.32    # kg
    inertial = np.array([0.003686, 0.003686, 0.006824])


    # # 1. 质量随机化 (例如上下浮动 15%)
    # mass_factor = np.random.uniform(0.5, 3)
    # drone.mass = mass * mass_factor
    
    # # 2. 转动惯量随机化 (通常各轴独立扰动 10%)
    # # 惯量必须保持正定，所以建议使用乘法扰动
    # inertia_factors = np.random.uniform(0.5, 2.0, size=3)
    # current_inertia = inertial * inertia_factors
    # drone.inertial = np.diag(current_inertia)
    # drone.inertial_inv = np.linalg.inv(drone.inertial)
    
    start_timestamp = time.perf_counter()
    # pdb.set_trace()
    while True:  # one rollout
        # 获取当前时刻的期望轨迹的信息
        state_des = desired_trajectory.get_desired_trajectory(drone.state.time)

        action = agent.get_action(obs, deterministic=False)
        next_obs = obs

        # 仿真姿态环，确保obs是位置环的频率。
        while not obs_flag:
            att = next_obs[6:9] + state_des.att
            ang = next_obs[9:] + state_des.ang
            scale = np.array([6, 6, 13])    # 13 = 1.32*9.8
            act_att = scale * np.array([action[0], action[1], action[2]+1])
            act_att = controller_att.NFC_att(act_att, att, ang, state_des)
            next_obs, reward, terminated, truncated, info = drone.step(act_att, state_des)
            obs_flag = info["obs_flag"]
                
        buffer.push(obs, action, reward, next_obs, terminated, truncated)      # 将数据存入缓冲区
        obs = next_obs
        obs_flag = False

        steps += 1
        episode_len += 1
        episode_reward += reward
        
        # policy update  <<<
        if len(buffer) > batch_size:
            loss_log = agent.update(buffer.sample(batch_size))
            for key, value in loss_log.items():
                logger.add_scalar(key, value, steps)

        if terminated or truncated:
            break
    
    episodes += 1

    returns.append(episode_reward)
    average_return = np.array(returns).mean() if len(returns) <= average_range else np.array(returns[-(average_range+1):-1]).mean()

    # verbose
    np.set_printoptions(precision=3, suppress=True, floatmode='fixed', linewidth=150)
    print(f"ID: {timestamp} | Steps: {steps} | Episodes: {episodes} | Length: {episode_len} | Reward: {round(episode_reward, 3)} | Average Return: {round(average_return, 3)} | Costed Real Time: {round(time.perf_counter() - start_timestamp, 3)} | OBS: {obs} | ACT: {action}")
    
    # logging
    logger.add_scalar('episodic/return', episode_reward, steps)
    logger.add_scalar('episodic/length', episode_len, steps)
    logger.add_scalar('episodic/return(average)', average_return, steps)
    logger.add_scalar('episode_nums', episodes, steps)


    # save model
    agent.save_model(f"{logger.get_logdir()}/ckpts/latest/")

    if episodes % model_save_interval == 0 or (len(returns) > average_return and average_return >= return_threshold):
        agent.save_model(f"{logger.get_logdir()}/ckpts/{episodes}/")
    
    if len(returns) > average_return and average_return >= return_threshold:
        print(f"Training SUCCESSFUL!")
        break