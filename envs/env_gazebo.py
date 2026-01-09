import sys
import os
sys.path.append('../')
sys.path.append('/home/magic/AdapRL_zh/RL_MRAC_RAL')

import numpy as np
import math
import copy
import gymnasium as gym
from scipy.spatial.transform import Rotation as R
from typing import Optional

from collections import defaultdict
from envs.utils import *
from envs.desired_trajectory import Desired_trajectory
from envs.controller import Controller
from envs.env_model import DroneEnv as DroneEnv_model

import pandas as pd


# ! /usr/bin/env python
import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from mavros_msgs.msg import State, AttitudeTarget, PositionTarget, State, Altitude, VFR_HUD

from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
from mavros_msgs.srv import CommandBool, SetMode
import numpy as np
import tty, termios, time, sys, select, PyKDL, math
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, BatteryState
from std_msgs.msg import Int16, Float32
from sensor_msgs.msg import Range


class DroneEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # 无人机系统参数
        self.mass = 1.32    # kg
        self.g = 9.81
        self.inertial = np.diag([0.003686, 0.003686, 0.006824])
        self.inertial_inv = np.linalg.inv(self.inertial)


        self.duration = 30.0     # 仿真时长
        position_frequency = 20.0
        self.dt = 1.0 / position_frequency  # 控制采样间隔
        hovering_throttle = 0.2     # 悬停油门
        self.POTT = hovering_throttle / (self.mass * self.g)     # 无人机的油门与推力的比例系数

        # 设置状态约束边界
        self.DEG2RAD = math.pi / 180        # 0.017453292519943295
        self.RAD2DEG = 180 / math.pi        # 57.2957795131
        self.MAX_ATT = 55 * self.DEG2RAD    # 最大姿态角约束
        self.MAX_ACC = 1.0 * self.g


        self.log_flag = True    # 是否记录log数据
        self.arm_state = False
        self.reset()

        # 初始化发送给px4-mixer的信息
        self.att_mixer = AttitudeTarget()

        #############
        rospy.init_node("M0_node")

        # Setpoint publishing MUST be faster than 10Hz
        self.rate = rospy.Rate(position_frequency)

        # pub
        self.set_state = rospy.Publisher("gazebo/set_model_state", ModelState, queue_size=10)
        self.target_motion_pub = rospy.Publisher("/mavros/setpoint_raw/local", PositionTarget, queue_size=10)
        self.body_target_pub = rospy.Publisher("/mavros/setpoint_raw/attitude", AttitudeTarget, queue_size=10)
        self.laser_pub = rospy.Publisher("/mavros/altitude", Altitude, queue_size=2)
        rospy.wait_for_service("/mavros/set_mode")
        self.flightModeService = rospy.ServiceProxy("/mavros/set_mode", SetMode)
        rospy.wait_for_service("/mavros/cmd/arming")
        self.armService = rospy.ServiceProxy("/mavros/cmd/arming", CommandBool)

        # subscribers
        # gazebo中用这个获取pos
        self.local_pose_sub = rospy.Subscriber("/mavros/local_position/pose", PoseStamped, self.mocap_pose_callback)
        # 真机中用这个获取pos
        # self.mocap_pose_sub = rospy.Subscriber("/mavros/vision_pose/pose", PoseStamped, self.mocap_pose_callback)

        self.local_vel_sub = rospy.Subscriber("/mavros/local_position/velocity_body", TwistStamped, self.local_vel_callback)
        self.odom_sub = rospy.Subscriber("/mavros/local_position/odom", Odometry, self.odom_callback)
        self.mavros_sub = rospy.Subscriber("/mavros/state", State, self.mavros_state_callback)
        self.laser_sub = rospy.Subscriber("/mavros/local_position/pose", PoseStamped, self.laser_callback)
        self.vfr_sub = rospy.Subscriber("/mavros/vfr_hud", VFR_HUD, self.vfr_callback)
        self.local_acc_sub = rospy.Subscriber("/mavros/imu/data", Imu, self.local_acc_callback)


        # 预留一些时间来等待arm实现。
        for _ in range(40):
            self.rate.sleep()

        # 等待 MAVROS 与飞控连接
        while (not rospy.is_shutdown() and not self.mavros_state.connected):
            self.rate.sleep()
        print('Connected!')

        ##########################################################
        # 先解锁UAV
        self.arm()
        print('Armed! Please take off firstly!')

        # 预留一些时间来等待arm实现。
        for _ in range(20):
            self.rate.sleep()




    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None, state: Optional[State_struct]=None):
        super().reset(seed=seed)

        # define initial state
        self.state = State_struct(
                    pos=np.array([0.0, 0.0, 0.0]),
                    vel=np.array([0.0, 0.0, 0.0]),
                    att=np.array([0.0, 0.0, 0.0]),
                    ang=np.array([0.0, 0.0, 0.0]),
                    time=0.0)
        self.pos = np.array([0.0, 0.0, 0.0])
        self.vel = np.array([0.0, 0.0, 0.0])
        self.acc = np.array([0.0, 0.0, 0.0])
        self.att = np.array([0.0, 0.0, 0.0])
        self.ang = np.array([0.0, 0.0, 0.0])

        self.time_start = time.time()
        self.seq = 0
        self.state.time = time.time() - self.time_start

        self.obs = State_struct()

        observation = np.concatenate([self.obs.pos, 
                                      self.obs.vel,
                                      self.obs.att,
                                      self.obs.ang], axis=0)
        return observation, self.obs



    

    def step(self, action, state_des):
        self.state.time += self.dt
        self.seq += 1

        self.throttle = action[0] * self.POTT
        self.att_des = action[1:]


        # publish 角速度命令
        if self.seq == 1:
            self.flightModeService(custom_mode='OFFBOARD')
        self.att_pub()
        self.rate.sleep()

        self.state.pos = self.pos.copy()
        self.state.vel = self.vel.copy()
        yaw, pitch, roll = R.from_quat(self.quat).as_euler('zyx')
        self.state.att = np.array([roll, pitch, yaw])
        self.state.ang = self.ang.copy()

        self.obs.get_error(self.state, state_des)

        observation = np.concatenate([self.obs.pos, 
                                      self.obs.vel,
                                      self.obs.att,
                                      self.obs.ang], axis=0)
        
        reward = 0.0

        terminated = False  # only current reward
        truncated = self.state.time > self.duration  # with future reward

        info = {"obs": self.obs, 
                "obs_flag": True }

        return observation, reward, terminated, truncated, info
        


    def close(self):
        self.flightModeService(custom_mode='AUTO.LAND')




    def mocap_pose_callback(self, msg):
        self.pos[0] = msg.pose.position.x
        self.pos[1] = msg.pose.position.y
        self.pos[2] = msg.pose.position.z
        x = msg.pose.orientation.x
        y = msg.pose.orientation.y
        z = msg.pose.orientation.z
        w = msg.pose.orientation.w
        self.quat = np.array([x, y, z, w])


    def local_vel_callback(self, msg):
        self.vel[0] = msg.twist.linear.x
        self.vel[1] = msg.twist.linear.y
        self.vel[2] = msg.twist.linear.z

    def local_acc_callback(self, msg):
        self.acc[0] = msg.linear_acceleration.x
        self.acc[1] = msg.linear_acceleration.y
        self.acc[2] = msg.linear_acceleration.z
        self.ang[0] = msg.angular_velocity.x
        self.ang[1] = msg.angular_velocity.y
        self.ang[2] = msg.angular_velocity.z

    def odom_callback(self, msg):
        x = msg.pose.pose.orientation.x
        y = msg.pose.pose.orientation.y
        z = msg.pose.pose.orientation.z
        w = msg.pose.pose.orientation.w
        rot = PyKDL.Rotation.Quaternion(x, y, z, w)
        self.att = np.array(rot.GetRPY())  # 显式转换为NumPy数组，替换原对象

        # self.roll_rate = msg.twist.twist.angular.x
        # self.pitch_rate = msg.twist.twist.angular.y
        # self.yaw_rate = msg.twist.twist.angular.z

    def laser_callback(self, msg):
        self.altitude = msg.pose.position.z

    def vfr_callback(self, msg):
        self.throttle_vfr = msg.throttle

    def mavros_state_callback(self, msg):
        self.mavros_state = msg
        self.arm_state = msg.armed

    def arm(self):
        if self.armService(True):
            self.arm_state = True
            return True
        else:
            print("Vehicle arming failed!")
            return False

    def disarm(self):
        if self.armService(False):
            print('disarm')
            return True
        else:
            print("Vehicle disarming failed!")
            return False



    def ang_pub(self):
        ang = self.ang_des
        self.att_mixer.body_rate.x = ang[0]
        self.att_mixer.body_rate.y = ang[1]
        self.att_mixer.body_rate.z = ang[2]
        self.att_mixer.thrust = self.throttle
        self.att_mixer.type_mask = AttitudeTarget.IGNORE_ATTITUDE
        self.body_target_pub.publish(self.att_mixer)

    def att_pub(self):
        qx, qy, qz, qw = euler_to_quaternion(self.att_des)
        self.att_mixer.orientation.w = qw
        self.att_mixer.orientation.x = qx
        self.att_mixer.orientation.y = qy
        self.att_mixer.orientation.z = qz
        self.att_mixer.thrust = self.throttle
        print(f'Throttle is {self.throttle}.')
        self.att_mixer.type_mask = AttitudeTarget.IGNORE_ROLL_RATE + AttitudeTarget.IGNORE_PITCH_RATE + AttitudeTarget.IGNORE_YAW_RATE
        self.body_target_pub.publish(self.att_mixer)
            

