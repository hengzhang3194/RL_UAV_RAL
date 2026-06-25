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

# ===========================================================================
# 2. 训练主程序
# ===========================================================================
if __name__ == "__main__":
    # 配置初始化
    config = DroneConfig.get_model(model_name='P600', duration=30.0)  
    raw_env = DroneEnv_model(config=config)
    desired_trajectory = Desired_trajectory(trajectory_flag='horizon_eight')
    controller_att = Controller_Attitude(controller_flag='NFC_att', config=config)

    # 实例化包装器环境
    env = DroneSB3Wrapper(raw_env, config, desired_trajectory, controller_att)

    # 确定计算硬件设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 日志初始化 (适配你原本的 TensorBoard 目录架构)
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_dir = f"./tensorboard/Drone_model/SB3_SAC/{timestamp}"
    new_logger = configure(log_dir, ["stdout", "tensorboard"])

    # 核心：定义 SB3 的 SAC 模型
    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,          # 对应你的 q_lr, pi_lr
        buffer_size=int(1e6),        # 对应你的 buffer_size
        learning_starts=256,         # 达到 batch_size 后开始训练
        batch_size=256,              # 对应你的 batch_size
        tau=0.005,                   # 对应你的 tau
        gamma=0.99,                  # 对应你的 gamma
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], qf=[256, 256])), # 对应你的 hidden_dims
        device=device,
        verbose=1
    )
    model.set_logger(new_logger)

    # 回调函数：定期保存 checkpoint
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,             # 每 10000 步保存一次模型
        save_path=os.path.join(log_dir, "ckpts"),
        name_prefix="sac_drone_model"
    )

    # 开启一键训练
    print("Starting training via Stable-Baselines3...")
    model.learn(
        total_timesteps=int(6e6),    # 对应你的 max_steps
        callback=checkpoint_callback,
        log_interval=10              # 每 10 个 episode 输出一次日志
    )

    # 保存最终模型
    model.save(os.path.join(log_dir, "ckpts", "latest"))
    print("Training finished and model saved.")