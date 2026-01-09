import numpy as np
import math
import copy
import gymnasium as gym
from scipy.spatial.transform import Rotation as R
from typing import List, Optional

from arcpy import arcpy
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
# Class of uav model in Flare.
######################################
class DroneEnv(gym.Env):

    def __init__(self, localhost: int = 25556):
        super().__init__()
        self.flare = arcpy.Controller('localhost', localhost)
        self.px4 = self.flare.get_object_by_path('Drone/core/px4')
        self.body_info = self.flare.get_object_by_path('Drone/core/info')


        # 无人机系统参数
        self.mass = 1.32    # kg
        self.g = 9.81
        self.inertial = np.diag([0.003686, 0.003686, 0.006824])
        self.inertial_inv = np.linalg.inv(self.inertial)

        self.duration = 30.0     # 仿真时长
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

        #######################################
        # 让无人机从空中起飞
        # Start simulation
        self.flare.start()
        self.flare.reset()
        self.sim_dt = self.flare.get_time_step()

        # 初始化之后，让无人机起飞并悬停
        steps_per_call = round(self.dt / self.sim_dt)
        nan = float('NaN')
        input = (0.0, 0.0, 0.0, 0.4, nan, nan, nan, nan, nan, nan, 0.0, nan)
        for i in range(100):
            reply = self.flare.simulate(steps_per_call,  {self.px4: input}, [self.body_info])
            print(f'Takeoff {i}/100.')
        #######################################


        self.log_flag = True    # 是否记录log数据

        self.observation_space = convert_observation_to_space(self.reset()[0])
        self.action_space      = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)  # avoid the open range caused by 'tanh' in SAC algorithm

        
        


    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None, state: Optional[State_struct]=None):
        super().reset(seed=seed)

        # define initial state
        self.state = State_struct(
                    pos=np.array([0.0, 0.0, 1.0]),
                    vel=np.array([0.0, 0.0, 0.0]),
                    att=np.array([0.0, 0.0, 0.0]),
                    ang=np.array([0.0, 0.0, 0.0]),
                    time=0.0)

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
        poscost = 10 / (np.linalg.norm(obs.pos) + 1)
        velcost = 1 / (np.linalg.norm(obs.vel) + 1)

        cost = yawcost + poscost + velcost

        return cost
  

    def step(self, action, state_des):

        self.state.time += self.dt
        self.seq += 1
        steps_per_call = round((self.state.time - self.sim_time) / self.sim_dt)
        self.sim_time += steps_per_call * self.sim_dt

        # Simulate a control period, giving action (dict), and requesting output (list).
        self.throttle = action[0] * self.POTT
        self.tau_roll = action[1]
        self.tau_pitch = action[2]
        self.tau_yaw = action[3]
        input = (3.0, self.tau_roll, self.tau_pitch, self.tau_yaw, self.throttle)
        reply = self.flare.simulate(steps_per_call,  {self.px4: input}, [self.body_info])

        # Check simulation result
        if reply.is_failed():
            raise ValueError('Flare Simulation failed!')
        else:
            # Get body info output (a tuple of floats)
            body_state = np.array(reply.get_output_of(self.body_info))
            pos = body_state[0:3]
            att = quaternion_to_euler(body_state[3:7])    # wxyz
            vel = body_state[7:10]
            ang = body_state[10:13]       # (rad/s)
            acc = body_state[13:16]

        self.state.pos = pos
        self.state.vel = vel
        self.state.att = att
        self.state.ang = ang
        self.state.acc = acc


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


        terminated = False  # only current reward
        truncated = self.state.time >= self.duration  # with future reward
         # 检查是否超出阈值
        threshold = 3.0  # 设定阈值，比如 3.0
        if np.linalg.norm(self.obs.pos) > threshold:
            terminated = True  # 终止当前 episode

        info = {"obs": self.obs, 
                "obs_flag": obs_flag }

        return observation, reward, terminated, truncated, info
    
    
    def close(self):
        # Clear simulator
        print("Finished, clearing...")
        self.flare.clear()

        # Close the simulation controller
        self.flare.close()
        print("Done.")
