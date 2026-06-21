import os
import numpy as np
import math, time
from scipy.linalg import solve_continuous_lyapunov
from scipy.integrate import solve_ivp
import pandas as pd

import torch
from torch.distributions import Normal
from envs.utils import *
from envs.controller_att import Controller_Attitude
from envs.sac_agent import ContinuousPolicyNetwork

import pdb

class Controller:

    def __init__(self, controller_flag='MRAC', config=None):
        # 无人机系统参数
        self.mass = config.mass    # 1.32 kg
        self.g = config.g
        self.inertial = config.inertial
        self.inertial_inv = np.linalg.inv(self.inertial)

        self.duration = config.duration     # 仿真时长
        self.pos_att_power = config.pos_att_power
        self.dt = config.dt  # 控制采样间隔
        self.dt_pos = config.dt_pos
        self.POTT = config.POTT

   

        self.DEG2RAD = math.pi / 180  # 0.01745
        self.RAD2DEG = 180 / math.pi     # 57.2958

        # Constraints
        self.MAX_ATT = 55 * self.DEG2RAD   # degree -> rad
        self.MAX_FORCE = 1.0 * self.g * self.mass


        self.force_controller = np.zeros(3)  # 控制器的位置环输出，惯性坐标系下XYZ方向的推力
        self.thrust = 0.0     # 机体坐标系下的Z-axis推力
        self.throttle = 0.0     # throttle command （0-1）
        self.torque_controller = np.zeros(3)  # 控制器的姿态环输出力矩


        # 选择姿态环控制器，并获取所选控制器的参数
        self.controller_flag = controller_flag
        self.controller_att = Controller_Attitude(controller_flag='NFC_att', config=config)
        self.get_controller_parameters()



    #################################
    # 获取控制器参数                 #
    #################################
    def get_controller_parameters(self):
        '''
        需要设置controller_flag参数，来选择使用哪一个控制器。
        - 'NFC+NFC': position-loop is NFC, attitude-loop is Nonlinear Feedback Controller.
        - 'MRAC+NFC': position-loop is MRAC, attitude-loop is Nonlinear Feedback Controller.
        - 'RL+NFC': position-loop is RL, attitude-loop is Nonlinear Feedback Controller.
        - 'RL-DR+NFC': position-loop is RL with domain randomization, attitude-loop is Nonlinear Feedback Controller.
        - 'MRAC_full_model': full model is MRAC.
        '''
        if self.controller_flag in ['NFC_model', 'NFC_flare']:
            # position loop 参数
            self.s_pos = np.zeros(3)       # auxiliary variable
            self.sum_s_pos = np.zeros(3)
            self.Gamma_pos = 5
            self.K_pos = np.array([2, 2, 10]) * 0.5
            # self.K_pos = np.array([50, 50, 20]) * 0.5
            self.Ki_pos = np.array([5, 5, 5]) * 0.2

        elif self.controller_flag == 'NFC_gazebo':
            # position loop 参数
            self.s_pos = np.zeros(3)       # auxiliary variable
            self.sum_s_pos = np.zeros(3)
            self.Gamma_pos = 5
            self.K_pos = np.array([2, 2, 10]) * 0.5
            self.Ki_pos = np.array([5, 5, 5]) * 0.2

        elif self.controller_flag == 'MRAC':
            # system matrix of reference model
            self.Am_k = 3
            self.A = np.block([
                [np.zeros((3, 3)), np.eye(3)],   # 第一行
                [np.zeros((3, 3)), np.zeros((3, 3))]  # 第二行
                ])
            self.B = np.block([
                [np.zeros((3, 3))],   # 第一行
                [np.eye(3) / self.mass]  # 第二行
                ])
            self.C = np.block([
                np.eye(3), np.zeros((3, 3))])
            self.Am = np.block([
                [np.zeros((3, 3)), np.eye(3)],   # 第一行
                [-self.Am_k * np.eye(3) / self.mass, -self.Am_k * np.eye(3) / self.mass]  # 第二行
                ])
            
            # state of reference model
            self.pos_ref = np.zeros(3)
            self.vel_ref = np.zeros(3)
            
            # 位置环采取MRAC，姿态环采取非线性反馈控制
            self.kx = np.zeros((3, 6))
            self.kr = np.zeros((3, 3))
            self.theta = np.zeros((6, 3))  
            self.gamma_x = 0.005 * np.eye(6)
            self.gamma_r = 0.003 * np.eye(3)
            self.gamma_theta = 0.003 * np.eye(6)
            self.Q = -800 * np.eye(6)  # Q 必须是对称的
            # 求解 Lyapunov 方程
            self.P = solve_continuous_lyapunov(self.Am.T, self.Q)

            ############ 从文件中读取控制器参数的初值
            # data_gain = np.load('Data/controller_gains.npz')
            # self.kx = data_gain['kx'][-1].reshape(3, 6)
            # self.kr = data_gain['kr'][-1].reshape(3, 3)
            # self.theta = data_gain['theta'][-1].reshape(6, 3)


        

        elif self.controller_flag in ['RL_model', 'RL_flare', 'RL_gazebo']:
            # 位置环 RL，姿态环采取非线性反馈控制
            obs_dims = 6
            act_dims = 3
            hidden_size=[256, 256]

            # pi_model_path = 'tensorboard/Drone_model/SAC/20260306_132832/ckpts/7800/pi.pth'
            pi_model_path = 'tensorboard/Drone_model/SAC/20260616_230603/ckpts/2200/pi.pth'

            assert os.path.exists(pi_model_path), f"Path '{pi_model_path}' of policy model DOESN'T exist."

            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.pi_net = ContinuousPolicyNetwork(obs_dims, act_dims, hidden_size).to(self.device)
            self.pi_net.load_state_dict(torch.load(pi_model_path))
            self.pi_net.eval()  # 切换到eval模式，让NN的预测更稳定。

        elif self.controller_flag == 'RL_corl':
            # 位置环 RL，姿态环采取非线性反馈控制
            obs_dims = 6
            act_dims = 3
            hidden_size=[256, 256]

            pi_model_path = 'tensorboard/rl_8h/pi.pth'

            assert os.path.exists(pi_model_path), f"Path '{pi_model_path}' of policy model DOESN'T exist."

            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.pi_net = ContinuousPolicyNetwork(obs_dims, act_dims, hidden_size).to(self.device)
            self.pi_net.load_state_dict(torch.load(pi_model_path))
            self.pi_net.eval()  # 切换到eval模式，让NN的预测更稳定。

        elif self.controller_flag == 'RL_full_model':
            # 位置环 RL，姿态环采取非线性反馈控制
            obs_dims = 12
            act_dims = 4
            hidden_size=[256, 256]
            pi_model_path = 'tensorboard/Drone_model/SAC/20251223_230017_full_model/ckpts/latest/pi.pth'
            assert os.path.exists(pi_model_path), f"Path '{pi_model_path}' of policy model DOESN'T exist."

            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.pi_net = ContinuousPolicyNetwork(obs_dims, act_dims, hidden_size).to(self.device)
            self.pi_net.load_state_dict(torch.load(pi_model_path))
            self.pi_net.eval()  # 切换到eval模式，让NN的预测更稳定。


        elif self.controller_flag == 'RL_by_flare':
            # 位置环 RL，姿态环采取非线性反馈控制
            obs_dims = 15
            act_dims = 3
            hidden_size=[256, 256]

            pi_model_path = 'tensorboard/Drone_flare/SAC/20260115_181221/ckpts/latest/pi.pth' 

            assert os.path.exists(pi_model_path), f"Path '{pi_model_path}' of policy model DOESN'T exist."

            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.pi_net = ContinuousPolicyNetwork(obs_dims, act_dims, hidden_size).to(self.device)
            self.pi_net.load_state_dict(torch.load(pi_model_path))
            self.pi_net.eval()  # 切换到eval模式，让NN的预测更稳定。


        elif self.controller_flag == 'RL_MRAC_flare':
            # system matrix of reference model
            self.Am_k = 3.0
            self.B = np.block([[np.zeros((3, 3))], 
                               [np.eye(3) / self.mass]])
            self.Am = np.block([[np.zeros((3, 3)), np.eye(3)], 
                                [-self.Am_k * np.eye(3), -self.Am_k * np.eye(3)]])
            
            # 位置环采取MRAC，姿态环采取非线性反馈控制
            self.kx = np.zeros((3, 6))
            self.kr = np.zeros((3, 3))
            self.theta = np.zeros((10, 3))  

            self.gamma_x = 0.003 * np.eye(6)
            self.gamma_r = 0.003 * np.eye(3)
            self.gamma_theta = 0.003 * np.eye(10)
            self.Q = -900 * np.eye(6)  # Q 必须是对称的
            self.P = solve_continuous_lyapunov(self.Am.T, self.Q)

            ############ 从文件中读取控制器参数的初值
            # data_gain = np.load('Data/RL_MRAC_flare_01.npz')
            # self.kx = data_gain['kx'][-1].reshape(3, 6)
            # self.kr = data_gain['kr'][-1].reshape(3, 3)
            # self.theta = data_gain['theta'][-1].reshape(12, 3)

            # load RL data
            RL_data = pd.read_csv('Data/RL_data.csv')
            self.pos_ref_all = RL_data[['pos_xd', 'pos_yd', 'pos_zd']].to_numpy()
            self.vel_ref_all = RL_data[['vel_xd', 'vel_yd', 'vel_zd']].to_numpy()
            self.ref_input_all = RL_data[['force_x', 'force_y', 'force_z']].to_numpy()
            self.pos_ref = np.zeros(3)
            self.vel_ref = np.zeros(3)

        elif self.controller_flag == 'RL_MRAC_landing_flare':
            # system matrix of reference model
            self.Am_k = 3.0
            self.B = np.block([[np.zeros((3, 3))], 
                               [np.eye(3) / self.mass]])
            self.Am = np.block([[np.zeros((3, 3)), np.eye(3)], 
                                [-self.Am_k * np.eye(3), -self.Am_k * np.eye(3)]
                ])
            
            # 位置环采取MRAC，姿态环采取非线性反馈控制
            self.kx = np.zeros((3, 6))
            self.kr = np.zeros((3, 3))
            self.theta = np.zeros((6, 3))  

            self.gamma_x = 0.02 * np.eye(6)
            self.gamma_r = 0.003 * np.eye(3)
            self.gamma_theta = 0.03 * np.eye(6)
            self.Q = -900 * np.eye(6)  # Q 必须是对称的
            self.P = solve_continuous_lyapunov(self.Am.T, self.Q)

            # GEF 预测
            self.f_gef_estimate = 0.0
            # self.theta_gef = np.array([2.405, -1.928, 5.793, -0.561, -0.596, 1.512])
            self.theta_gef = np.zeros((6)) 

            ############ 从文件中读取控制器参数的初值
            data_gain = np.load('Data/test0.npz')
            self.kx = data_gain['kx'][-1].reshape(3, 6)
            self.kr = data_gain['kr'][-1].reshape(3, 3)
            self.theta = data_gain['theta'][-1].reshape(6, 3)

            # load RL data
            RL_data = pd.read_csv('Data/RL_data.csv')
            self.pos_ref_all = RL_data[['pos_xd', 'pos_yd', 'pos_zd']].to_numpy()
            self.vel_ref_all = RL_data[['vel_xd', 'vel_yd', 'vel_zd']].to_numpy()
            self.ref_input_all = RL_data[['force_x', 'force_y', 'force_z']].to_numpy()
            self.pos_ref = np.zeros(3)
            self.vel_ref = np.zeros(3)

        elif self.controller_flag == 'RL_MRAC_gazebo':
            # system matrix of reference model
            self.Am_k = 3.0
            self.B = np.block([
                [np.zeros((3, 3))],   # 第一行
                [np.eye(3) / self.mass]  # 第二行
                ])
            self.Am = np.block([
                [np.zeros((3, 3)), np.eye(3)],   # 第一行
                [-self.Am_k * np.eye(3), -self.Am_k * np.eye(3)]  # 第二行
                ])
            
            # 位置环采取MRAC，姿态环采取非线性反馈控制
            self.kx = np.zeros((3, 6))     
            self.kr = np.zeros((3, 3))
            self.theta = np.zeros((6, 3))  
            self.gamma_x = 0.003 * np.eye(6)
            self.gamma_r = 0.0003 * np.eye(3)
            self.gamma_theta = 0.003 * np.eye(6)
            self.Q = -900 * np.eye(6)  # Q 必须是对称的
            self.P = solve_continuous_lyapunov(self.Am.T, self.Q)

            # GEF 预测
            self.f_gef_estimate = 0.0
            # self.theta_gef = np.array([2.405, -1.928, 5.793, -0.561, -0.596, 1.512])
            self.theta_gef = np.zeros((6)) 

            ############ 从文件中读取控制器参数的初值
            # data_gain = np.load('Data/test.npz')
            # self.kx = data_gain['kx'][-1].reshape(3, 6)
            # self.kr = data_gain['kr'][-1].reshape(3, 3)
            # self.theta = data_gain['theta'][-1].reshape(6, 3)

            # load RL data
            RL_data = pd.read_csv('Data/RL_data.csv')
            self.pos_ref_all = RL_data[['pos_xd', 'pos_yd', 'pos_zd']].to_numpy()
            self.vel_ref_all = RL_data[['vel_xd', 'vel_yd', 'vel_zd']].to_numpy()
            self.ref_input_all = RL_data[['force_x', 'force_y', 'force_z']].to_numpy()
            self.pos_ref = np.zeros(3)
            self.vel_ref = np.zeros(3)

        elif self.controller_flag == 'RL_MRAC_landing_gazebo':
            # system matrix of reference model
            self.Am_k = 3.0
            self.B = np.block([
                [np.zeros((3, 3))],   # 第一行
                [np.eye(3) / self.mass]  # 第二行
                ])
            self.Am = np.block([
                [np.zeros((3, 3)), np.eye(3)],   # 第一行
                [-self.Am_k * np.eye(3), -self.Am_k * np.eye(3)]  # 第二行
                ])
            
            # 位置环采取MRAC，姿态环采取非线性反馈控制
            self.kx = np.zeros((3, 6))     
            self.kr = np.zeros((3, 3))
            self.theta = np.zeros((6, 3))  
            self.gamma_x = 0.003 * np.eye(6)
            self.gamma_r = 0.0003 * np.eye(3)
            self.gamma_theta = 0.003 * np.eye(6)
            self.Q = -900 * np.eye(6)  # Q 必须是对称的
            self.P = solve_continuous_lyapunov(self.Am.T, self.Q)

            ############ 从文件中读取控制器参数的初值
            # data_gain = np.load('Data/RL_MRAC_flare.npz')
            # self.kx = data_gain['kx'][-1].reshape(3, 6)
            # self.kr = data_gain['kr'][-1].reshape(3, 3)
            # self.theta = data_gain['theta'][-1].reshape(6, 3)

            # load RL data
            RL_data = pd.read_csv('Data/RL_data.csv')
            self.pos_ref_all = RL_data[['pos_xd', 'pos_yd', 'pos_zd']].to_numpy()
            self.vel_ref_all = RL_data[['vel_xd', 'vel_yd', 'vel_zd']].to_numpy()
            self.ref_input_all = RL_data[['force_x', 'force_y', 'force_z']].to_numpy()
            self.pos_ref = np.zeros(3)
            self.vel_ref = np.zeros(3)


        elif self.controller_flag == 'MRAC_full_model':
            # 位置环采取MRAC，姿态环采取非线性反馈控制
            self.kx = np.zeros((4, 12))
            self.kr = np.zeros((4, 4))
            self.theta = np.zeros((12, 4))  

            # 系统线性化矩阵
            self.A = np.zeros((12, 12))
            self.A[0:6, 6:12] = np.eye(6)
            temp_A = np.array([[0, self.g, 0], [-self.g, 0, 0], [0, 0, 0]])
            self.A[6:9, 3:6] = temp_A
            self.B = np.zeros((12, 4))
            self.B[8, 0] = 1 / self.mass
            self.B[9:12, 1:4] = self.J_inv

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

            # mrac 参数
            self.gamma_x = 0.05 * np.eye(12)
            self.gamma_r = 0.003 * np.eye(4)
            self.gamma_theta = 0.03 * np.eye(12)
            self.Q = -1200 * np.eye(12)  # Q 必须是对称的
            # 求解 Lyapunov 方程
            self.P = solve_continuous_lyapunov(self.Am.T, self.Q)



    #####################################
    # NFC for model
    #####################################
    def NFC_model(self, obs_flag, state_des):
        if obs_flag:
            ### Translation control
            self.pos_error = self.pos - self.pos_des
            self.vel_error = self.vel - self.vel_des

            self.s_pos = self.vel_error + self.Gamma_pos * self.pos_error
            self.sum_s_pos = self.sum_s_pos + self.s_pos * self.dt_pos # 将采样时间从姿态环变成位置环

            self.M = self.mass * np.eye(3)
            self.G = self.mass * np.array([0, 0, self.g])
            
            self.vel_ref = self.vel_des - self.Gamma_pos * self.pos_error
            self.acc_ref = self.acc_des - self.Gamma_pos * self.vel_error

            thrust = np.dot(self.M, self.acc_ref) + self.G - self.K_pos * self.s_pos - self.Ki_pos * self.sum_s_pos - 3*self.mass * self.pos - 3*self.mass * self.vel

            # Control constrain
            thrust = np.clip(thrust, [-self.MAX_FORCE, -self.MAX_FORCE, 0.0], [self.MAX_FORCE, self.MAX_FORCE, 1.8 * self.MAX_FORCE])

            # pdb.set_trace()
            self.force_controller = thrust

        action_pos = self.force_controller

        # 计算姿态环控制器
        action_att = self.controller_att.get_controller(action_pos, self.att, self.ang, self.att, self.ang)
        action = np.concatenate([action_pos, action_att], axis=0)

        return action

    
    #####################################
    # NFC : for Flare
    #####################################
    def NFC_flare(self, obs_flag, state_des):
        if obs_flag:
            ### Translation control
            self.pos_error = self.pos - self.pos_des
            self.vel_error = self.vel - self.vel_des

            self.s_pos = self.vel_error + self.Gamma_pos * self.pos_error
            self.sum_s_pos = self.sum_s_pos + self.s_pos * self.dt_pos # 将采样时间从姿态环变成位置环

            self.M = self.mass * np.eye(3)
            self.G = self.mass * np.array([0, 0, self.g])
            
            self.vel_ref = self.vel_des - self.Gamma_pos * self.pos_error
            self.acc_ref = self.acc_des - self.Gamma_pos * self.vel_error

            thrust = np.dot(self.M, self.acc_ref) + self.G - self.K_pos * self.s_pos - self.Ki_pos * self.sum_s_pos

            # Control constrain
            thrust = np.clip(thrust, [-self.MAX_FORCE, -self.MAX_FORCE, 0.0], [self.MAX_FORCE, self.MAX_FORCE, 1.8 * self.MAX_FORCE])

            # pdb.set_trace()
            self.force_controller = thrust

        action_pos = self.force_controller

        # 计算姿态环控制器
        action_att = self.controller_att.get_controller(action_pos, self.att, self.ang, self.att_des, self.ang_des)

        action = np.concatenate([action_pos, action_att], axis=0)

        return action
    
    #####################################
    # NFC : for Gazebo
    #####################################
    def NFC_gazebo(self, obs_flag, state_des):

        ### Translation control
        self.pos_error = self.pos - self.pos_des
        self.vel_error = self.vel - self.vel_des

        self.s_pos = self.vel_error + self.Gamma_pos * self.pos_error
        self.sum_s_pos += self.s_pos * self.dt_pos # 将采样时间从姿态环变成位置环

        self.M = self.mass * np.eye(3)
        self.G = self.mass * np.array([0, 0, self.g])
        
        self.vel_ref = self.vel_des - self.Gamma_pos * self.pos_error
        self.acc_ref = self.acc_des - self.Gamma_pos * self.vel_error

        thrust = np.dot(self.M, self.acc_ref) + self.G - self.K_pos * self.s_pos - self.Ki_pos * self.sum_s_pos

        # Control constrain
        thrust = np.clip(thrust, [-self.MAX_FORCE, -self.MAX_FORCE, 0.0], [self.MAX_FORCE, self.MAX_FORCE, 1.8 * self.MAX_FORCE])

        # pdb.set_trace()
        self.force_controller = thrust

        # 计算姿态环控制器
        thrust_body, att_des = self.controller_att.Decomposition1(thrust, state_des.att)

        # 返回值是：机身推力 + 3维期望姿态
        action = np.array([thrust_body, att_des[0], att_des[1], att_des[2]])

        return action




    #####################################
    # MRAC for Flare
    #####################################
    def mrac_controller(self, obs_flag, state_des):
        if obs_flag:
            ### Translation control
            self.pos_error = self.pos_ref - self.pos
            self.vel_error = self.vel_ref - self.vel
            error = np.hstack((self.pos_error, self.vel_error)).reshape(-1,1)
            state = np.hstack((self.pos, self.vel)).reshape(-1,1)
            ref_input = self.ref_input.reshape(-1,1)
            # phi = np.hstack((self.pos, self.vel)).reshape(-1,1)
            # phi = np.hstack((self.pos, self.vel, self.att, self.ang)).reshape(-1,1)

            
            # phi_m = self.acc_des    # 质量项 (惯性 + 重力)
            # phi_d_linear = self.vel     # 阻尼项 v, v*|v|
            # phi_d_quad =  self.vel * abs(self.vel)
            # phi_bias = 1.0    # 常值偏置项 (外力干扰)

            # # 水平拼接成 3x10 矩阵
            # phi = np.hstack([phi_m, phi_d_linear, phi_d_quad, phi_bias]).reshape(-1,1)

            Eh = self.calculate_Eh(self.pos[2])
            term1 = self.f_gef_estimate * self.throttle
            term2 = self.f_gef_estimate * self.pos[2]
            term3 = self.throttle * Eh
            term4 = (self.f_gef_estimate**2) * self.pos[2]
            term5 = (self.f_gef_estimate**2) * Eh
            term6 = (self.throttle**2) * Eh
            phi = np.array([term1, term2, term3, term4, term5, term6]).reshape(-1,1)

            self.theta_gef = np.array([2.405, -1.928, 5.793, -0.561, -0.596, 1.512])

            kx_update = self.gamma_x @ state @ error.T @ self.P @ self.B
            kr_update = self.gamma_r @ ref_input @ error.T @ self.P @ self.B
            theta_update = - self.gamma_theta @ phi @ error.T @ self.P @ self.B

            sigma = 0.001
            # self.kx += (kx_update.T - sigma * self.kx) * self.dt_pos
            # self.kr += (kr_update.T - sigma * self.kr) * self.dt_pos
            # self.theta += (theta_update - sigma * self.theta) * self.dt_pos

            self.kx = self.kx + (kx_update.T) * self.dt_pos
            self.kr = self.kr + (kr_update.T) * self.dt_pos
            self.theta = self.theta + (theta_update) * self.dt_pos
            # self.G = self.mass * np.array([0, 0, self.g]).reshape(-1, 1)
            # thrust = self.kx @ state + self.kr @ ref_input - self.theta.T @ phi + self.G - 3.0 * state[0:3] - 3.0 * state[3:6]
            
            # thrust = self.kx @ state + self.kr @ ref_input - self.theta.T @ phi + self.G

            # Sindy library
            theta_term = self.theta.T @ phi
            self.f_gef_estimate += theta_term[2][0] * self.dt_pos
            # theta_term = self.theta_gef @ phi
            # pdb.set_trace()
            # self.f_gef_estimate += theta_term[0] * self.dt_pos
            self.G = np.array([0, 0, self.mass * self.g - self.f_gef_estimate]).reshape(-1, 1)
            thrust = self.kx @ state + self.kr @ ref_input + self.G - 3.0 * state[0:3] - 3.0 * state[3:6]

            print(f'GEF is: {self.f_gef_estimate}.')

            # Control constrain
            thrust = np.clip(thrust, [-self.MAX_FORCE, -self.MAX_FORCE, 0.0], [self.MAX_FORCE, self.MAX_FORCE, 1.8 * self.MAX_FORCE])
            
            self.force_controller = thrust.flatten()

            # if self.time > 10.0:
            #     pdb.set_trace()

        action_pos = self.force_controller
        action_att = self.controller_att.get_controller(action_pos, self.att, self.ang, self.att_des, self.ang_des)
        self.throttle = action_att[0] * self.POTT

        action = np.concatenate([action_pos, action_att], axis=0)

        return action
    
    #####################################
    # MRAC for Gazebo
    #####################################
    def RL_MRAC_gazebo(self, obs_flag, state_des):
        ### Translation control
        self.pos_error = self.pos_ref - self.pos
        self.vel_error = self.vel_ref - self.vel
        error = np.hstack((self.pos_error, self.vel_error)).reshape(-1,1)
        state = np.hstack((self.pos, self.vel)).reshape(-1,1)
        ref_input = self.ref_input.reshape(-1,1)
        # phi = np.hstack((self.pos, self.vel)).reshape(-1,1)

        Eh = self.calculate_Eh(self.pos[2])
        term1 = self.f_gef_estimate * self.throttle
        term2 = self.f_gef_estimate * self.pos[2]
        term3 = self.throttle * Eh
        term4 = (self.f_gef_estimate**2) * self.pos[2]
        term5 = (self.f_gef_estimate**2) * Eh
        term6 = (self.throttle**2) * Eh
        phi = np.array([term1, term2, term3, term4, term5, term6]).reshape(-1,1)
        


        kx_update = self.gamma_x @ state @ error.T @ self.P @ self.B
        kr_update = self.gamma_r @ ref_input @ error.T @ self.P @ self.B
        theta_update = - self.gamma_theta @ phi @ error.T @ self.P @ self.B
        self.kx = self.kx + kx_update.T * self.dt_pos
        self.kr = self.kr + kr_update.T * self.dt_pos
        self.theta = self.theta + theta_update * self.dt_pos
        # self.G = self.mass * np.array([0, 0, self.g]).reshape(-1, 1)

        theta_term = self.theta.T @ phi
        self.f_gef_estimate += theta_term[2][0] * self.dt_pos
        self.G = np.array([0, 0, self.mass * self.g - self.f_gef_estimate]).reshape(-1, 1)

        # thrust = self.kx @ state + self.kr @ ref_input - self.theta.T @ phi + self.G  - self.Am_k * state[0:3] - self.Am_k * state[3:6]
        # thrust = self.kx @ state + self.kr @ ref_input - self.theta.T @ phi + self.G

        thrust = self.kx @ state + self.kr @ ref_input + self.G - 3.0 * state[0:3] - 3.0 * state[3:6]

        
        # Control constrain
        thrust = np.clip(thrust, [-self.MAX_FORCE, -self.MAX_FORCE, 0.0], [self.MAX_FORCE, self.MAX_FORCE, 1.8 * self.MAX_FORCE])

        self.force_controller = thrust.flatten()


        # 计算姿态环控制器
        thrust_body, att_des = self.controller_att.Decomposition1(self.force_controller, state_des.att)
        self.throttle = thrust_body * self.POTT

        # 返回值是：机身推力 + 3维期望姿态
        action = np.array([thrust_body, att_des[0], att_des[1], att_des[2]])

        return action


        



    #####################################
    # MRAC for Flare （full uav model）
    #####################################
    def mrac_full_model_for_flare(self, pos_control_flag):
        # 获取当前时刻的参考系统的参考输出
        self.get_reference_model()

        # 获取当前UAV的observation
        self.get_observation(pos_control_flag)

        # 定义误差，以及组合变量为state
        self.pos_error = self.pos_des - self.pos_obs
        self.vel_error = self.vel_des - self.vel_obs
        self.att_error = self.att_des - self.att_obs
        self.ang_error = self.ang_des - self.ang_obs
        error = np.hstack((self.pos_error, self.att_error, self.vel_error, self.ang_error)).reshape(-1,1)
        state = np.hstack((self.pos_obs, self.att_obs, self.vel_obs, self.ang_obs)).reshape(-1,1)
        phi = np.hstack((self.pos_obs, self.att_obs, self.vel_obs, self.ang_obs)).reshape(-1,1)

        kx_update = self.gamma_x @ state @ error.T @ self.P @ self.B
        kr_update = self.gamma_r @ self.rl_input @ error.T @ self.P @ self.B
        theta_update = - self.gamma_theta @ phi @ error.T @ self.P @ self.B
        self.kx = self.kx + kx_update.T * self.dt
        self.kr = self.kr + kr_update.T * self.dt
        self.theta = self.theta + theta_update * self.dt
        self.G = self.mass * np.array([self.g, 0, 0, 0]).reshape(-1, 1)
        pos = np.array(self.pos).reshape(-1, 1)
        vel = np.array(self.vel).reshape(-1, 1)

        input_MRAC = self.kx @ state + self.kr @ self.rl_input - self.theta.T @ phi - self.K @ state + self.G
        
        # Control constrain
        input_MRAC[0] = max(min(input_MRAC[0], 1.8 * self.MAX_ACC * self.mass), 0.0)
        input_MRAC[1:4] = np.clip(input_MRAC[1:4], -1.0, 1.0)

        self.throttle = input_MRAC[0] * self.POTT  # 将机体坐标系下的推力转化为油门（0-1）
        self.tau_roll = input_MRAC[1]
        self.tau_pitch = input_MRAC[2]
        self.tau_yaw = input_MRAC[3]


    #####################################
    # MRAC for model-only
    #####################################
    def mrac_controller_model(self, pos_control_flag):

        if pos_control_flag:
            ### Translation control
            self.pos_error = self.pos_des - self.pos
            self.vel_error = self.vel_des - self.vel
            error = np.hstack((self.pos_error, self.vel_error)).reshape(-1,1)
            state = np.hstack((self.pos, self.vel)).reshape(-1,1)
            state_ref = self.ref.reshape(-1,1)
            phi = np.hstack((self.pos, self.vel)).reshape(-1,1)

            kx_update = self.gamma_x @ state @ error.T @ self.P @ self.B
            kr_update = self.gamma_r @ state_ref @ error.T @ self.P @ self.B
            theta_update = - self.gamma_theta @ phi @ error.T @ self.P @ self.B
            self.kx = self.kx + kx_update.T * self.dt_pos
            self.kr = self.kr + kr_update.T * self.dt_pos
            self.theta = self.theta + theta_update * self.dt_pos
            self.G = self.mass * np.array([0, 0, self.g]).reshape(-1, 1)
            pos = np.array(self.pos).reshape(-1, 1)
            vel = np.array(self.vel).reshape(-1, 1)

            thrust = self.kx @ state + self.kr @ state_ref - self.theta.T @ phi
            
            # Control constrain
            thrust = np.clip(thrust, [-self.MAX_FORCE, -self.MAX_FORCE, 0.0], [self.MAX_FORCE, self.MAX_FORCE, 1.8 * self.MAX_FORCE])

            self.force_controller = thrust.flatten()


    def get_reference_model(self):
        state_ref = np.hstack((self.pos_des, self.att_des, self.vel_des, self.ang_des)).reshape(-1,1)
        state_ref_update = self.Am @ state_ref + self.B @ self.rl_input
        state_ref = state_ref + state_ref_update * self.dt

        self.pos_des = state_ref[0:3].flatten()
        self.att_des = state_ref[3:6].flatten()
        self.vel_des = state_ref[6:9].flatten()
        self.ang_des = state_ref[9:12].flatten()

        

    #####################################
    # RL for Model (pos: linear, att: nonlinear)
    #####################################
    def RL_model(self, obs_flag, state_des):
        if obs_flag:
            # obs = np.hstack((self.pos_error, self.vel_error, self.att_error, self.ang_error))
            obs = np.hstack((self.pos_error, self.vel_error))
            obs_tensor = torch.FloatTensor(obs).to(self.device)
            mean, log_std = self.pi_net(obs_tensor)
            action = torch.tanh(mean).detach().cpu().numpy()
                
            # RL controller 处理
            force_scale = np.array([0.5, 0.5, 1.0]) * self.mass * self.g
            self.force_controller = action * force_scale

        action_pos = self.force_controller + self.mass * np.array([0, 0, self.g]) - 3.0 * self.pos - 3.0 * self.vel
        action_att = self.controller_att.get_controller(action_pos, self.att, self.ang, self.att_des, self.ang_des)
        action = np.concatenate([action_pos, action_att], axis=0)

        return action
    
    #####################################
    # RL for Model & Flare
    #####################################
    def RL_flare(self, obs_flag, state_des):
        if obs_flag:
            # obs = np.hstack((self.pos_error, self.vel_error, self.att_error, self.ang_error))
            obs = np.hstack((self.pos_error, self.vel_error))
            obs_tensor = torch.FloatTensor(obs).to(self.device)
            mean, log_std = self.pi_net(obs_tensor)
            action = torch.tanh(mean).detach().cpu().numpy()
                
            # RL controller 处理
            force_scale = np.array([0.5, 0.5, 1.0]) * self.mass * self.g
            self.force_controller = action * force_scale

        action_pos = self.force_controller + self.mass * np.array([0, 0, self.g]) - 3.0 * self.pos - 3.0 * self.vel
        action_att = self.controller_att.get_controller(action_pos, self.att, self.ang, self.att_des, self.ang_des)
        action = np.concatenate([action_pos, action_att], axis=0)

        return action
    
    #####################################
    # RL for Gazebo
    #####################################
    def RL_gazebo(self, obs_flag, state_des):
        obs = np.hstack((self.pos_error, self.vel_error))
        obs_tensor = torch.FloatTensor(obs).to(self.device)
        mean, log_std = self.pi_net(obs_tensor)
        action = torch.tanh(mean).detach().cpu().numpy()
            
        # RL controller 处理
        action[2] += 1.0
        force_scale = np.array([0.5, 0.5, 1.0]) * self.mass * self.g
        self.force_controller = action * force_scale

        action_pos = self.force_controller - 3.0 * self.pos - 3.0 * self.vel
        thrust_body, att_des = self.controller_att.Decomposition1(action_pos, state_des.att)

        # 返回值是：机身推力 + 3维期望姿态
        action = np.array([thrust_body, att_des[0], att_des[1], att_des[2]])

        return action
    
    #####################################
    # RL for Model & Flare
    #####################################
    def RL_corl(self, obs_flag, state_des):
        if obs_flag:
            self.pos_error = self.pos_des - self.pos
            self.vel_error = self.vel_des - self.vel
            obs = np.hstack((self.pos_error, self.vel_error))
            obs_tensor = torch.FloatTensor(obs).to(self.device)
            mean, log_std = self.pi_net(obs_tensor)
            action = torch.tanh(mean).detach().cpu().numpy()

            scale_act = action * np.array([3, 3, 6])

            self.G = self.mass * np.array([0, 0, self.g])

            thrust = scale_act + (self.G  - 3*self.pos - 3*self.vel).flatten()
            
            # Control constrain
            thrust[0] = max(min(thrust[0], 9.8 * self.mass), -9.8 * self.mass)
            thrust[1] = max(min(thrust[1], 9.8 * self.mass), -9.8 * self.mass)
            thrust[2] = max(min(thrust[2], 1.8 * 9.8 * self.mass), 0.0)

            self.force_controller = thrust.flatten()
            
        action_pos = self.force_controller

        # 计算姿态环控制器
        action_att = self.controller_att.NFC_att(action_pos, self.att, self.ang, state_des)
        # action = np.concatenate([action_pos, action_att], axis=0)
        action = action_att

        return action
    
    #####################################
    # RL for Model & Flare
    #####################################
    def RL_by_flare(self, obs_flag, state_des):
        if obs_flag:
            obs = np.hstack((self.pos_error, self.vel_error, self.att_error, self.ang_error, self.att))
            obs_tensor = torch.FloatTensor(obs).to(self.device)
            mean, log_std = self.pi_net(obs_tensor)
            std = log_std.exp()
            
            action = torch.tanh(mean).detach().cpu().numpy()
                
            # RL controller 处理
            self.force_controller[0] = action[0] * 0.5 * self.mass * self.g
            self.force_controller[1] = action[1] * 0.5 * self.mass * self.g
            self.force_controller[2] = (action[2]+1) * self.mass * self.g
            
        action_pos = self.force_controller

        # 计算姿态环控制器
        action_att = self.controller_att.get_controller(action_pos, self.att, self.ang, self.att_des, self.ang_des)
        action = action_att

        return action
    
    

    
    #####################################
    # RL with full model (4 input, 12 state) for Model
    #####################################
    def RL_full_model(self, obs_flag):
        # if obs_flag:
        obs = 1 * np.hstack((self.pos_error, self.vel_error, self.att_error, self.ang_error))
        obs_tensor = torch.FloatTensor(obs).to(self.device)
        mean, log_std = self.pi_net(obs_tensor)
        action = torch.tanh(mean).detach().cpu().numpy()
            
        # RL controller 处理
        scale = np.array([13, 1, 1, 1])    # 13 = 1.32*9.8
        action = scale * np.array([action[0], action[1], action[2], action[3]])  # [Thrust, Torque_x, Torque_y, Torque_z]

        return action
    


    #####################################
    # RL Control for model-only
    #####################################
    def RL_controller_model(self, pos_control_flag):
        if pos_control_flag:
            pos = np.array(self.pos).reshape(-1, 1)
            vel = np.array(self.vel).reshape(-1, 1)
            self.G = self.mass * np.array([0, 0, self.g]).reshape(-1, 1)

            self.pos_error = self.pos_des - self.pos
            self.vel_error = self.vel_des - self.vel
            obs = np.hstack((self.pos_error, self.vel_error))
            act = self.rl_controller.get_action_pi(obs).reshape(-1, 1)
            scale_act = act * np.array([[6], [6], [13]])

            thrust = scale_act

            self.force_controller = thrust.flatten()

        
        


    
    ###################################################
    # reference_model 的状态更新
    ###################################################
    def reference_model_update(self):
        state = np.hstack((self.pos_ref, self.vel_ref)).reshape(-1,1)
        input = self.ref_input.reshape(-1,1)
        state_update = self.Am @ state + self.B @ input
        next_state = state + state_update * self.dt
        self.pos_ref, self.vel_ref = next_state.reshape(2, 3)

    
    def calculate_Eh(self, z):
        """
        计算 Sanchez-Cuevas 系数 E_h，其中
        - R 是桨叶半径
        - d 是对角轴距
        - b 是机体宽度
        对于 P600 而言, R,d,b = 0.19, 0.45, 0.6 [m]
        对于 M0 而言, R,d,b = 0.06, 0.25, 0.6 [m]
        """
        R = 0.19
        d = 0.45
        b = 0.6
        Kb = 0.5
        
        z_safe = max(z, 0.05)   # 防止 z 过小导致分母为 0.
        term1 = 1 - (R / (4 * z_safe)) ** 2 - R ** 2 * (z_safe / np.sqrt((d ** 2 + 4 * z_safe ** 2) ** 3))
        term2 = (R ** 2 / 2) * (z_safe / np.sqrt((2 * d ** 2 + 4 * z_safe ** 2) ** 3))
        term3 = 2 * R ** 2 * (z_safe / np.sqrt((b ** 2 + 4 * z_safe ** 2) ** 3)) * Kb
        
        eh = (np.power(term1 - term2 - term3, -1) - 1)
        return eh













    ###########################################
    # Select controller by self.controller_flag
    ###########################################
    def get_controller(self, state, state_des_old, state_des, obs_flag):
        '''
        和上面选择控制参数的名称一样，调用相关的控制器
        注意传入的state其实是state_error。
        '''
        self.pos_error = state.pos
        self.vel_error = state.vel
        self.acc_error = state.acc
        self.att_error = state.att
        self.ang_error = state.ang
        self.time = state.time

        self.pos_des = state_des.pos
        self.vel_des = state_des.vel
        self.acc_des = state_des.acc
        self.att_des = state_des.att
        self.ang_des = state_des.ang

        self.pos = self.pos_error + state_des_old.pos
        self.vel = self.vel_error + state_des_old.vel
        self.acc = self.acc_error + state_des_old.acc
        self.att = self.att_error + state_des_old.att
        self.ang = self.ang_error + state_des_old.ang

        # NFC 相关
        if self.controller_flag == 'NFC_flare':
            action = self.NFC_flare(obs_flag, state_des)
        elif self.controller_flag == 'NFC_model':
            action = self.NFC_model(obs_flag, state_des)
        elif self.controller_flag == 'NFC_gazebo':
            action = self.NFC_gazebo(obs_flag, state_des)

        # MRAC 相关
        elif self.controller_flag == 'MRAC':
            self.ref_input = self.pos_des
            self.reference_model_update()
            action = self.mrac_controller(obs_flag, state_des)  

        # RL_MRAC 相关
        elif self.controller_flag == 'RL_MRAC_flare':
            index = round(self.time / self.dt)
            self.pos_ref = self.pos_ref_all[index]
            self.vel_ref = self.vel_ref_all[index]
            self.ref_input = self.ref_input_all[index]
            action = self.mrac_controller(obs_flag, state_des)  
        elif self.controller_flag == 'RL_MRAC_gazebo':
            index = round(self.time / self.dt)
            self.pos_ref = self.pos_ref_all[index]
            self.vel_ref = self.vel_ref_all[index]
            self.ref_input = self.ref_input_all[index]
            action = self.RL_MRAC_gazebo(obs_flag, state_des) 
        elif self.controller_flag == 'RL_MRAC_landing_flare':
            index = round(self.time / self.dt)
            self.pos_ref = self.pos_ref_all[index]
            self.vel_ref = self.vel_ref_all[index]
            self.ref_input = self.ref_input_all[index]
            action = self.mrac_controller(obs_flag, state_des)  

        # RL 相关
        elif self.controller_flag == 'RL_model':
            action = self.RL_model(obs_flag, state_des)
        elif self.controller_flag == 'RL_flare':
            action = self.RL_flare(obs_flag, state_des)
        elif self.controller_flag == 'RL_gazebo':
            action = self.RL_gazebo(obs_flag, state_des)
        elif self.controller_flag == 'RL_corl':
            action = self.RL_corl(obs_flag, state_des)
        elif self.controller_flag == 'RL_full_model':
            action = self.RL_full_model(obs_flag)
        elif self.controller_flag == 'RL_by_flare':
            action = self.RL_by_flare(obs_flag, state_des)

        else:
            raise ValueError(
                f"无效的控制器标识: {self.controller_flag}!") 

        return action






    

 