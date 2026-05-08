import numpy as np
import math, time
from scipy.linalg import solve_continuous_lyapunov
from scipy.integrate import solve_ivp

from envs.utils import *

import pdb


class Controller_Attitude:
    '''
    输入：3维力（世界坐标系）
    输出：1维力（机体坐标系）+3维力矩
    '''

    def __init__(self, controller_flag='NFC_att', config=None):
        # 无人机系统参数
        self.mass = config.mass    # 1.32 kg
        self.g = config.g
        self.inertial = config.inertial
        self.inertial_inv = np.linalg.inv(self.inertial)

        self.duration = config.duration     # 仿真时长
        self.pos_att_power = config.pos_att_power
        self.dt = config.dt  # 控制采样间隔
        
        
        
        self.DEG2RAD = math.pi / 180  # 0.01745
        self.RAD2DEG = 180 / math.pi     # 57.2958

        # Constraints
        self.MAX_ATT = 55 * self.DEG2RAD   # degree -> rad


        # 获取所选控制器的参数
        self.controller_flag = controller_flag
        self.get_controller_parameters()

    def get_controller_parameters(self):
        '''
        需要设置controller_flag参数，来选择使用哪一个控制器。
        - 'NFC_att': attitude-loop is Nonlinear Feedback Controller.
        - 'PID': attitude-loop is PID Controller.
        '''
        if self.controller_flag == 'NFC_att':
            # 姿态环 参数
            self.s_att = np.zeros(3)       # auxiliary variable
            self.sum_s_att = np.zeros(3)
            self.sum_att_error = np.zeros(3)
            self.Gamma_att = 3
            self.KP_ATT = np.array([5, 5, 5]) * 0.4
            self.KD_ATT = np.array([2, 2, 2]) * 0.2
            self.KI_ATT = np.array([1, 1, 1]) * 0.2

            self.temp_ang = np.zeros(3) # 存储当前的3维“姿态速度”，以方便计算“姿态加速度”。

        elif self.controller_flag == 'px4_att':

            # --- 姿态环参数 (Angle P Loop) ---
            self.KP_ATT = np.array([6.5, 6.5, 2.8]) # 典型PX4增益
            
            # --- 角速度环参数 (Rate PID Loop) ---
            self.KP_RATE = np.array([0.15, 0.15, 0.2])
            self.KI_RATE = np.array([0.05, 0.05, 0.1])
            self.KD_RATE = np.array([0.003, 0.003, 0.0])
            
            # 状态累计
            self.rate_integral = np.zeros(3)
            self.last_ang_error = np.zeros(3)

            self.temp_ang = np.zeros(3) # 存储当前的3维“姿态速度”，以方便计算“姿态加速度”。


    def NFC_att(self, att, ang, att_des, ang_des):
        '''att is attitude, ang is attitude velocity'''

        att_acc_des = (ang_des - self.temp_ang) / self.dt 
        self.temp_ang = ang_des

        att_error = att - att_des
        ang_error = ang - ang_des

        s_att = ang_error + self.Gamma_att * att_error
        self.sum_s_att += s_att * self.dt

        att_acc_ref = att_acc_des - self.Gamma_att * ang_error
        self.sum_att_error += att_error * self.dt

        torque_controller = np.dot(self.inertial, att_acc_ref) + np.cross(ang, np.dot(self.inertial, np.array(ang))) - self.KP_ATT * att_error - self.KD_ATT * ang_error - self.KI_ATT * self.sum_att_error

        # Control constrain
        max_torque = 1.0
        torque_controller = np.clip(torque_controller, -max_torque, max_torque)

        return torque_controller
    
    def px4_att(self, force, att, ang, att_des, ang_des):
        '''att is attitude, ang is attitude velocity'''

        # 1. 姿态环 (Outer Loop: P Control)
        # 计算欧拉角偏差 (注意：PX4底层实际用四元数，这里用欧拉角简化模拟)
        att_error = att_des - att
        # 这里的输出是期望角速度 (Rate Setpoint)
        ang_des = self.KP_ATT * att_error
        
        # 2. 角速度环 (Inner Loop: PID Control)
        ang_error = ang_des - ang
        
        # 积分项 (带抗饱和抗积分风暴)
        self.rate_integral += ang_error * self.dt
        self.rate_integral = np.clip(self.rate_integral, -0.3, 0.3)
        
        # 微分项
        ang_deriv = (ang_error - self.last_ang_error) / self.dt
        self.last_ang_error = ang_error
        
        # 计算归一化力矩输出 (PID)
        # 注意：Gazebo底层通常先计算一个无量纲的控制量，再乘以增益
        torque_controller = (self.KP_RATE * ang_error + 
                         self.KI_RATE * self.rate_integral + 
                         self.KD_RATE * ang_deriv)
        
        # Control constrain
        max_torque = 1.0
        torque_controller = np.clip(torque_controller, -max_torque, max_torque)
        
        return torque_controller
    

    def Decomposition1(self, force, att_des):
        ddx, ddy, ddz = force / self.mass

        # controller得到的Fz已经考虑了g了，这里不需要重复考虑
        thrust_body = self.mass * np.sqrt(ddx ** 2 + ddy ** 2 + ddz ** 2)

        # 角度的单位是（rad）
        yaw_des = att_des[2]

        pitch_den = ddx * np.cos(yaw_des) + ddy * np.sin(yaw_des)
        pitch_num = ddz
        pitch_des = np.arctan(pitch_den / pitch_num) if pitch_num != 0 else 0

        roll_den = np.sin(pitch_des) * (ddx * np.sin(yaw_des) - ddy * np.cos(yaw_des))
        roll_num = ddx * np.cos(yaw_des) + ddy * np.sin(yaw_des)
        roll_des = np.arctan(roll_den / roll_num) if roll_num != 0 else 0

        # 将解耦得到的姿态和期望姿态叠加
        att_decom = np.array([roll_des, pitch_des, yaw_des])
        att_des = np.clip(att_decom + att_des, -self.MAX_ATT, self.MAX_ATT)
        
        return thrust_body, att_des
    

    ###########################################
    # Select controller by self.controller_flag
    ###########################################
    def get_controller(self, force, att, ang, att_des, ang_des):
        '''
        和上面self.controller_flag的名称一样，调用相关的控制器
        # att is system attitude
        # ang is system attitude rate
        '''

        thrust_body, att_des = self.Decomposition1(force, att_des)

        if self.controller_flag == 'NFC_att':
            torque_controller = self.NFC_att(att, ang, att_des, ang_des)
        elif self.controller_flag == 'px4_att':
            torque_controller = self.NFC_att(att, ang, att_des, ang_des)
        
        else:
            raise ValueError(
                f"无效的控制器标识: {self.controller_flag}!") 
        
        tau_roll, tau_pitch, tau_yaw = torque_controller
        action = np.array([thrust_body, tau_roll, tau_pitch, tau_yaw])

        return action


    
