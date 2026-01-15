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
# Class of uav dynamic nonlinear model
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
        self.actuator_tau = 0.02 # 电机时间常数 [s]
        self.action_last = np.zeros(4)

        # 设置状态约束边界
        self.DEG2RAD = math.pi / 180        # 0.017453292519943295
        self.RAD2DEG = 180 / math.pi        # 57.2957795131
        self.MAX_ATT = 55 * self.DEG2RAD    # 最大姿态角约束
        self.MAX_ACC = 1.0 * self.g


        self.log_flag = True    # 是否记录log数据
        
        # self.reset()
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

        self.actual_action = np.zeros(4)
        
        self.obs = State_struct()
        self.obs_last = State_struct()

        observation = np.concatenate([self.obs.pos, 
                                      self.obs.vel,
                                      self.obs.att,
                                      self.obs.ang,
                                      self.state.att], axis=0)
        return observation, self.obs




    def reward(self, obs, action, last_action):
        yawcost = -0.5 * np.abs(obs.att[2])
        poscost = 10 / (np.linalg.norm(obs.pos) + 1)
        velcost = 1 / (np.linalg.norm(obs.vel) + 1)

        # 惩罚当前动作与上一个动作的差异，迫使控制逻辑变得平滑
        action_diff = np.linalg.norm(action - last_action)
        action_smoothness_cost = -0.5 * action_diff  # 系数 0.5 可根据震荡剧烈程度调整
        
        # 惩罚过大的力输出，避免系统总是在极限边缘运行
        action_mag_cost = -0.1 * np.linalg.norm(action)

        cost = yawcost + poscost + velcost + action_smoothness_cost + action_mag_cost

        return cost


    def step(self, action, state_des):

        self.state.time += self.dt
        self.seq += 1


        self.throttle = action[0] * self.POTT
        self.tau_roll = action[1]
        self.tau_pitch = action[2]
        self.tau_yaw = action[3]

        # 考虑控制器时延
        alpha = min(self.dt / self.actuator_tau, 1.0)
        self.actual_action = self.actual_action + alpha * (action - self.actual_action)
        action = self.actual_action

        

        # force
        state = np.concatenate([self.state.pos, self.state.vel, self.state.att, self.state.ang], axis=0)
        next_state = self._integrate_dynamics(state, action)

        (self.state.pos, self.state.vel, self.state.att, self.state.ang) = next_state.reshape(4, 3)


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
                                      self.obs.ang,
                                      self.state.att], axis=0)


        reward = self.reward(self.obs, action, self.action_last)
        self.action_last = action

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
            vel = s[3:6]
            att = s[6:9]
            ang = s[9:12]

            # 姿态部分
            phi, theta, psi = att
            W = np.array([
                [1, np.sin(phi) * np.tan(theta), np.cos(phi) * np.tan(theta)],
                [0, np.cos(phi), -np.sin(phi)],
                [0, np.sin(phi) / np.cos(theta), np.cos(phi) / np.cos(theta)]])
            dot_att = W @ ang
            dot_ang = self.inertial_inv @ (input[1:] - np.cross(ang, self.inertial @ ang))

            # 位置部分
            Rz = np.array([
                [np.cos(psi), -np.sin(psi), 0],
                [np.sin(psi), np.cos(psi), 0],
                [0, 0, 1]])
            Ry = np.array([
                [np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)]])
            Rx = np.array([
                [1, 0, 0],
                [0, np.cos(phi), -np.sin(phi)],
                [0, np.sin(phi), np.cos(phi)]])
            R = Rz @ Ry @ Rx
            thrust_world = R @ np.array([0, 0, input[0]])
            dot_vel = (thrust_world + np.array([0, 0, -self.mass * self.g])) / self.mass
            
            return np.concatenate([vel, dot_vel, dot_att, dot_ang])

        # RK4 步进
        k1 = dot_state(state, action)
        k2 = dot_state(state + dt/2 * k1, action)
        k3 = dot_state(state + dt/2 * k2, action)
        k4 = dot_state(state + dt * k3, action)
        
        next_state = state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

        return next_state