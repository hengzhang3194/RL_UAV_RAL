import numpy as np
import math
import copy
import gymnasium as gym
from scipy.spatial.transform import Rotation as R
from typing import Optional

from envs.utils import *

import pdb



def convert_observation_to_space(observation):
    '''
    This is copied from offical codes of 'gym'.
    '''
    from collections import OrderedDict
    if isinstance(observation, dict):
        space = gym.spaces.Dict(OrderedDict([(key, convert_observation_to_space(value)) for key, value in observation.items()]))
    elif isinstance(observation, np.ndarray):
        low   = np.full(observation.shape, -float("inf"), dtype=np.float32)
        high  = np.full(observation.shape, float("inf"), dtype=np.float32)
        space = gym.spaces.Box(low, high, dtype=observation.dtype)
    else:
        raise NotImplementedError(type(observation), observation)
    return space


######################################
# Class of uav dynamic linear model 
######################################
class DroneEnv(gym.Env):
    def __init__(self, localhost: int = 25556):
        super().__init__()


        # 无人机系统参数
        self.mass = 1.32    # kg
        self.g = 9.81
        self.inertial = np.diag([0.003686, 0.003686, 0.006824])
        self.inertial_inv = np.linalg.inv(self.inertial)


        self.duration = 300.0     # 仿真时长
        position_frequency = 20.0
        attitude_frequency = 200.0
        self.pos_att_power = round(attitude_frequency / position_frequency)
        self.dt = 1.0 / attitude_frequency  # 控制采样间隔
        hovering_throttle = 0.4     # 悬停油门
        self.POTT = hovering_throttle / (self.mass * self.g)     # 无人机的油门与推力的比例系数

        # 设置状态约束边界
        self.DEG2RAD = math.pi / 180        # 0.017453292519943295
        self.RAD2DEG = 180 / math.pi        # 57.2957795131
        self.MAX_ATT = 55 * self.DEG2RAD    # 最大姿态角约束
        self.MAX_ACC = 1.0 * self.g

        # system matrix
        self.Am_k = 3
        self.A = np.block([
            [np.zeros((3, 3)), np.eye(3)],   # 第一行
            [np.zeros((3, 3)), np.zeros((3, 3))]  # 第二行
            ])
        self.B = np.block([
            [np.zeros((3, 3))],   # 第一行
            [np.eye(3) / self.mass]  # 第二行
            ])
        self.Am = np.block([
            [np.zeros((3, 3)), np.eye(3)],   # 第一行
            [-self.Am_k * np.eye(3), -self.Am_k * np.eye(3)]  # 第二行
            ])


        self.log_flag = True    # 是否记录log数据
        
        # self.observation_space = convert_observation_to_space(self.reset()[0])
        self.reset()
        self.observation_space = gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(6,), dtype=np.float32)
        self.action_space      = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)  # avoid the open range caused by 'tanh' in SAC algorithm
        # self.action_space = gym.spaces.Box(low=np.array([-9.8, -20, -20, -2]), high=np.array([30, 20, 20, 2]))

        
        


    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None, state: Optional[State_struct]=None):
        super().reset(seed=seed)

        # define initial state
        self.state = State_struct(
                    pos=np.array([0.0, 0.0, 0.0]),
                    vel=np.array([0.0, 0.0, 0.0]),
                    att=np.array([0.0, 0.0, 0.0]),
                    ang=np.array([0.0, 0.0, 0.0]),
                    time=0.0)
        
        # initialize time
        self.state.time = 0.0
        self.sim_time = 0.0     # flare仿真器上一次的时间
        self.seq = 0
        
        self.obs = State_struct()
        self.obs_last = State_struct()

        observation = np.concatenate([self.obs.pos, 
                                      self.obs.vel,
                                      self.obs.att,
                                      self.obs.ang], axis=0)
        return observation, self.obs




    def reward(self, obs):
        yawcost = 0.5 * np.abs(obs.att[2])

        # err_xy = np.linalg.norm(obs.pos[:2])
        # err_z = np.abs(obs.pos[2])
        # poscost = 10 / (err_xy + 1) + 20 / (err_z + 1)
        poscost = 20 / (np.linalg.norm(obs.pos) + 1)
        velcost = 1 / (np.linalg.norm(obs.vel) + 1)
        z_cost = 5 * np.exp(-np.abs(obs.pos[2]))

        cost = yawcost + poscost + velcost + z_cost

        return cost
  

    def step(self, action, state_des):

        self.state.time += self.dt
        self.seq += 1

        action_pos = action[0:3]
        self.throttle = action[3] * self.POTT
        self.tau_roll = action[4]
        self.tau_pitch = action[5]
        self.tau_yaw = action[6]



        # force
        state_pos = np.concatenate([self.state.pos, self.state.vel], axis=0)
        state_update = self.Am @ state_pos + self.B @ action_pos
        next_state = state_pos + state_update * self.dt 
        (self.state.pos, self.state.vel) = next_state.reshape(2, 3)


        state_att = np.concatenate([self.state.att, self.state.ang], axis=0)
        next_state = self._integrate_dynamics(state_att, action[4:7])
        (self.state.att, self.state.ang) = next_state.reshape(2, 3)

        


        # 根据控制频率更新obs的值
        obs_flag = self.seq % self.pos_att_power == 1
        if obs_flag:
            self.obs.get_error(self.state, state_des)
            self.obs_last.get_error(self.state, state_des)
        else:
            self.obs.get_error(self.state, state_des)
            self.obs.pos = self.obs_last.pos
            self.obs.vel = self.obs_last.vel
            self.obs.acc = self.obs_last.acc

        # MDP (Markov Decision Process)
        observation = np.concatenate([self.obs.pos, 
                                      self.obs.vel,
                                      self.obs.att,
                                      self.obs.ang], axis=0)


        reward = self.reward(self.obs)

        # conditions of termination
        terminated = False  # only current reward
        truncated = self.state.time > self.duration  # with future reward
         # 检查是否超出阈值
        threshold = 3.0  # 设定阈值，比如 3.0
        if np.linalg.norm(self.obs.pos) > threshold:
            # reward -= 10000  # 给予较大的负 reward
            terminated = True  # 终止当前 episode

        info = {"obs": self.obs, 
                "obs_flag": obs_flag }

        return observation, reward, terminated, truncated, info
    

    def close(self):
        pass


    def _integrate_dynamics(self, state, action):
        """内部调用的 RK4 积分器"""
        dt = self.dt
        
        # 定义导数闭包，确保使用的是当前步的控制输入
        def dot_state(s, input):
            # 解包中间状态
            att = s[0:3]
            ang = s[3:6]

            # 姿态部分
            phi, theta, psi = att
            W = np.array([
                [1, np.sin(phi) * np.tan(theta), np.cos(phi) * np.tan(theta)],
                [0, np.cos(phi), -np.sin(phi)],
                [0, np.sin(phi) / np.cos(theta), np.cos(phi) / np.cos(theta)]])
            dot_att = W @ ang
            dot_ang = self.inertial_inv @ (input - np.cross(ang, self.inertial @ ang))
            
            return np.concatenate([dot_att, dot_ang])

        # RK4 步进
        k1 = dot_state(state, action)
        k2 = dot_state(state + dt/2 * k1, action)
        k3 = dot_state(state + dt/2 * k2, action)
        k4 = dot_state(state + dt * k3, action)
        
        next_state = state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

        return next_state