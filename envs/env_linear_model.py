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


        self.duration = 20.0     # 仿真时长
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
        self.A = np.zeros((12, 12))
        self.A[0:6, 6:12] = np.eye(6)
        self.A[6:9, 3:6] = np.array([[0, self.g, 0], [-self.g, 0, 0], [0, 0, 0]])
        self.B = np.zeros((12, 4))
        self.B[8, 0] = 1 / self.mass
        self.B[9:12, 1:4] = self.inertial_inv

        # 期望极点（实部为负的复数），阻尼比为0.7
        p_pos = -1.0
        p_att = -6.0
        p_vel = -3.0
        p_ang = -10.0
        poles = np.array([p_pos, p_pos, p_pos,  # x, y, z
                        p_att, p_att, p_att,    # phi, theta, psi
                        p_vel, p_vel, p_vel,    # vx, vy, vz
                        p_ang, p_ang, p_ang    # p, q, r
                        ])

        # 使用 place_poles 函数求解 K
        from scipy.signal import place_poles
        pp = place_poles(self.A, self.B, poles)
        self.K = pp.gain_matrix
        self.Am = self.A - self.B @ self.K


        self.log_flag = True    # 是否记录log数据
        
        self.observation_space = convert_observation_to_space(self.reset()[0])
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

        cost = yawcost + poscost + velcost

        return cost
  

    def step(self, action, state_des):

        self.state.time += self.dt
        self.seq += 1


        self.throttle = action[0] * self.POTT
        self.tau_roll = action[1]
        self.tau_pitch = action[2]
        self.tau_yaw = action[3]


        # force
        state = np.concatenate([self.state.pos, self.state.att, self.state.vel, self.state.ang], axis=0)
        state += (self.A @ state + self.B @ action) * self.dt
        (self.state.pos, self.state.att, self.state.vel, self.state.ang) = state.reshape(4, 3)
        # self.state.pos = state[0:3]
        # self.state.att = state[3:6]
        # self.state.vel = state[6:9]
        # self.state.ang = state[9:12]

        


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

        if obs_flag:
            reward = self.reward(self.obs)
        else:   
            reward = 0.0

        # conditions of termination
        terminated = False  # only current reward
        # terminated = terminated or np.linalg.norm(self.desired_trajectory[-1][0] - self.pos) < self.position_error_threshold  # reach the goal position
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
