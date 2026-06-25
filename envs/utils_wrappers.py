import sys
import os
sys.path.append('../')

import numpy as np
import time
import gymnasium as gym
from gymnasium.spaces import Box

import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure

from envs.env_model import DroneEnv as DroneEnv_model
from envs.desired_trajectory import Desired_trajectory
from envs.controller_att import Controller_Attitude
from envs.model_configuration import DroneConfig

# ===========================================================================
# 1. 核心封装：自定义 SB3 兼容环境包装器
# ===========================================================================
class DroneSB3Wrapper(gym.Wrapper):
    def __init__(self, env, config, desired_trajectory, controller_att):
        super().__init__(env)
        self.config = config
        self.desired_trajectory = desired_trajectory
        self.controller_att = controller_att
        
        # SB3 的输入观测只取前 6 维 [pos_error, vel_error]
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        # SAC 核心动作空间：3维
        self.action_space = Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        
        # 用于缓存完整的 12 维状态向量
        self.current_full_obs = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # 把最原始的 12 维状态先存下来
        self.current_full_obs = obs.copy()
        # 提取前 6 维状态返回给 SB3 算法
        return obs[0:6].astype(np.float32), info

    def step(self, action):
        # 1. 获取当前时刻的期望轨迹信息
        state_des = self.desired_trajectory.get_desired_trajectory(self.env.state.time)
        
        # 2. 映射动作为实际物理期望推力
        act_pos = np.zeros(3)
        act_pos[0] = action[0] * 0.5 * self.env.mass * self.env.g
        act_pos[1] = action[1] * 0.5 * self.env.mass * self.env.g
        act_pos[2] = (action[2] + 1) * self.env.mass * self.env.g
        
        # 【修复核心】：直接使用上一时刻缓存的完整 12 维状态
        pos = self.current_full_obs[0:3] + state_des.pos
        vel = self.current_full_obs[3:6] + state_des.vel
        act_pos = act_pos - 3.0 * pos - 3.0 * vel

        cumulative_reward = 0.0
        obs_flag = False
        next_obs = self.current_full_obs.copy()

        # 3. 内部姿态环高频仿真模拟
        while not obs_flag:
            att = next_obs[6:9] + state_des.att
            ang = next_obs[9:12] + state_des.ang
            att_des = state_des.att
            ang_des = state_des.ang

            act_att = self.controller_att.get_controller(act_pos, att, ang, att_des, ang_des)
            act = np.concatenate([act_pos, act_att], axis=0)
            next_obs, reward, terminated, truncated, info = self.env.step(act, state_des)
            obs_flag = info["obs_flag"]
            cumulative_reward += reward

        # 【同步更新缓存】：把仿真走完之后的最新 12 维状态存起来供下一次 step 使用
        self.current_full_obs = next_obs.copy()

        # 返回前 6 维给算法网络
        return next_obs[0:6].astype(np.float32), cumulative_reward, terminated, truncated, info
