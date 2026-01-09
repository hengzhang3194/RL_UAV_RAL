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

    def __init__(self, controller_flag='MRAC+NFC'):
        # 无人机系统参数
        self.mass = 1.32    # kg
        self.inertial = np.diag([0.003686, 0.003686, 0.006824])
        self.inertial_inv = np.linalg.inv(self.inertial)
        self.g = 9.8
        hovering_throttle = 0.4     # 悬停油门
        self.POTT = hovering_throttle / (self.mass * self.g)     # 无人机的油门与推力的比例系数
        self.dt = 1 / 200
        
        

        self.DEG2RAD = math.pi / 180  # 0.01745
        self.RAD2DEG = 180 / math.pi     # 57.2958

        # Constraints
        self.MAX_ATT = 55 * self.DEG2RAD   # degree -> rad
        self.MAX_ACC = 9.8 * 1.0


        self.att_decom = np.zeros(3)    # 解耦之后的期望姿态
        self.temp_ang = np.zeros(3) # 存储当前的3维“姿态速度”，以方便计算“姿态加速度”。


        self.force_controller = np.zeros(3)  # 控制器的位置环输出，惯性坐标系下XYZ方向的推力
        self.thrust = 0.0     # 机体坐标系下的Z-axis推力
        self.throttle = 0.0     # throttle command （0-1）
        self.torque_controller = np.zeros(3)  # 控制器的姿态环输出力矩


        # 获取所选控制器的参数
        self.controller_flag = controller_flag
        self.controller_att = Controller_Attitude()
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
        if self.controller_flag == 'NFC':
            # position loop 参数
            self.s_pos = np.zeros(3)       # auxiliary variable
            self.sum_s_pos = np.zeros(3)
            self.Gamma_pos = 5
            self.K_pos = np.array([2, 2, 10]) * 0.5
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
                [-self.Am_k * np.eye(3), -self.Am_k * np.eye(3)]  # 第二行
                ])
            
            # state of reference model
            self.pos_ref = np.zeros(3)
            self.vel_ref = np.zeros(3)
            
            # 位置环采取MRAC，姿态环采取非线性反馈控制
            self.kx = np.zeros((3, 6))     
            self.kr = np.zeros((3, 3))
            self.theta = np.zeros((6, 3))  
            self.gamma_x = 0.005 * np.eye(6)
            self.gamma_r = 0.0003 * np.eye(3)
            self.gamma_theta = 0.0003 * np.eye(6)
            self.Q = -900 * np.eye(6)  # Q 必须是对称的
            # 求解 Lyapunov 方程
            self.P = solve_continuous_lyapunov(self.Am.T, self.Q)

            ############ 从文件中读取控制器参数的初值
            # data_gain = np.load(path+'.npz')
            # self.kx = data_gain['kx'][-1].reshape(3, 6)
            # self.kr = data_gain['kr'][-1].reshape(3, 3)
            # self.theta = data_gain['theta'][-1].reshape(6, 3)



        elif self.controller_flag == 'MRAC_regressor':
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
                [-self.Am_k * np.eye(3), -self.Am_k * np.eye(3)]  # 第二行
                ])
            
            # state of reference model
            self.pos_ref = np.zeros(3)
            self.vel_ref = np.zeros(3)
            
            # 位置环采取MRAC，姿态环采取非线性反馈控制
            self.kx = np.zeros((3, 6))     
            self.kr = np.zeros((3, 3))
            self.theta = np.zeros((6, 3))  
            self.gamma_x = 0.005 * np.eye(6)
            self.gamma_r = 0.0003 * np.eye(3)
            self.gamma_theta = 0.0003 * np.eye(6)
            self.Q = -900 * np.eye(6)  # Q 必须是对称的
            # 求解 Lyapunov 方程
            self.P = solve_continuous_lyapunov(self.Am.T, self.Q)

            ############ 从文件中读取控制器参数的初值
            # data_gain = np.load(path+'.npz')
            # self.kx = data_gain['kx'][-1].reshape(3, 6)
            # self.kr = data_gain['kr'][-1].reshape(3, 3)
            # self.theta = data_gain['theta'][-1].reshape(6, 3)


        elif self.controller_flag == 'RL_full_model':
            # 位置环 RL，姿态环采取非线性反馈控制
            obs_dims = 12
            act_dims = 4
            hidden_size=[256, 256]
            # pi_model_path = 'policy/20250710_000013/ckpts/latest/pi.pth'
            pi_model_path = 'tensorboard/Drone_model/SAC/20251223_230017/ckpts/1000/pi.pth'
            assert os.path.exists(pi_model_path), f"Path '{pi_model_path}' of policy model DOESN'T exist."

            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.pi_net = ContinuousPolicyNetwork(obs_dims, act_dims, hidden_size).to(self.device)
            self.pi_net.load_state_dict(torch.load(pi_model_path))
            self.pi_net.eval()  # 切换到eval模式，让NN的预测更稳定。

        elif self.controller_flag == 'RL':
            # 位置环 RL，姿态环采取非线性反馈控制
            obs_dims = 12
            act_dims = 3
            hidden_size=[256, 256]

            # pi_model_path = 'policy/rl_8h/pi.pth'
            pi_model_path = 'tensorboard/Drone_model/SAC/20260108_164139/ckpts/5000/pi.pth' 

            assert os.path.exists(pi_model_path), f"Path '{pi_model_path}' of policy model DOESN'T exist."

            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.pi_net = ContinuousPolicyNetwork(obs_dims, act_dims, hidden_size).to(self.device)
            self.pi_net.load_state_dict(torch.load(pi_model_path))
            self.pi_net.eval()  # 切换到eval模式，让NN的预测更稳定。

        elif self.controller_flag == 'RL_gazebo':
            # 位置环 RL，姿态环采取非线性反馈控制
            obs_dims = 12
            act_dims = 3
            hidden_size=[256, 256]

            # pi_model_path = 'policy/rl_8h/pi.pth'
            pi_model_path = 'policy/pi.pth'

            assert os.path.exists(pi_model_path), f"Path '{pi_model_path}' of policy model DOESN'T exist."

            self.device = 'cpu'
            self.pi_net = ContinuousPolicyNetwork(obs_dims, act_dims, hidden_size).to(self.device)
            self.pi_net.load_state_dict(torch.load(pi_model_path))
            self.pi_net.eval()  # 切换到eval模式，让NN的预测更稳定。




        elif self.controller_flag == 'RL-DR':
            # 位置环 RL random，姿态环采取非线性反馈控制，96dbf08c
            controllers = []
            controllers.append(Controller(name='drone_test/(v1)', ctrl_dt=0.05, action_range=1.0, pi_model_path='tensorboard/rl_random_8h/pi.pth', obs_dims=6, act_dims=3, hidden_size=[256, 256]))
            self.rl_controller = controllers[-1]



            data = self.load_trajectory_from_csv('Data/rl_data.csv')

            # 参考轨迹
            self.x_vals = np.array(data['/goal_position']['x'])
            self.y_vals = np.array(data['/goal_position']['y'])
            self.z_vals = np.array(data['/goal_position']['z'])
            
            self.vx_vals = np.array(data['/goal_velocity']['x'])
            self.vy_vals = np.array(data['/goal_velocity']['y'])
            self.vz_vals = np.array(data['/goal_velocity']['z'])

            self.ax_vals = np.array(data['/action']['x']) * 6.0
            self.ay_vals = np.array(data['/action']['y']) * 6.0
            self.az_vals = np.array(data['/action']['z']) * 13.0

        elif self.controller_flag == 'RL_MRAC':
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
                [-self.Am_k * np.eye(3), -self.Am_k * np.eye(3)]  # 第二行
                ])
            
            # 位置环采取MRAC，姿态环采取非线性反馈控制
            self.kx = np.zeros((3, 6))     
            self.kr = np.zeros((3, 3))
            self.theta = np.zeros((6, 3))  
            self.gamma_x = 0.005 * np.eye(6)
            self.gamma_r = 0.0003 * np.eye(3)
            self.gamma_theta = 0.0003 * np.eye(6)
            self.Q = -900 * np.eye(6)  # Q 必须是对称的
            self.P = solve_continuous_lyapunov(self.Am.T, self.Q)

            ############ 从文件中读取控制器参数的初值
            # data_gain = np.load('Data/controller_gains.npz')
            # self.kx = data_gain['kx'][-1].reshape(3, 6)
            # self.kr = data_gain['kr'][-1].reshape(3, 3)
            # self.theta = data_gain['theta'][-1].reshape(6, 3)

            # load RL data
            RL_data = pd.read_csv('Data/RL_data.csv')

            self.pos_ref_all = RL_data[['pos_x', 'pos_y', 'pos_z']].to_numpy()
            self.vel_ref_all = RL_data[['vel_x', 'vel_y', 'vel_z']].to_numpy()
            self.ref_input_all = RL_data[['force_x', 'force_y', 'force_z']].to_numpy()
            self.ref_time_all = RL_data['time'].to_numpy()


        elif self.controller_flag == 'MRAC_full_model':
            # 位置环采取MRAC，姿态环采取非线性反馈控制
            self.kx = np.zeros((4, 12))     
            self.kr = np.zeros((4, 4))
            self.theta = np.zeros((12, 4))  

            ############ 从文件中读取控制器参数的初值
            # data_gain = np.load(path+'.npz')
            # self.kx = data_gain['kx'][-1].reshape(3, 6)
            # self.kr = data_gain['kr'][-1].reshape(3, 3)
            # self.theta = data_gain['theta'][-1].reshape(6, 3)

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
        
            # 读取rl训练好的数据（参考信号和参考输出）
            # self.rl_data = load_csv_data('Data/rl_data.csv')

        elif self.controller_flag == 'NFC_regressor':
            # self.theta_hat = np.array([
            #     4.501,      # GEFz*throttle*exp(-floor_dis)
            #     -4.390,     # GEFz*pos_z*exp(-floor_dis)
            #     -1.625,     # GEFz^2*pos_z*exp(-floor_dis)
            #     -1500.061,  # throttle*Eh*exp(-floor_dis)
            #     -4.512,     # GEFz^2*Eh*exp(-floor_dis)
            #     3236.996,   # throttle^2*Eh*exp(-floor_dis)
            #     -3.669,     # GEFz*throttle
            #     3.367,      # GEFz*pos_z
            #     1.391,      # GEFz^2*pos_z
            #     1382.492,   # throttle*Eh
            #     4.465,      # GEFz^2*Eh
            #     -2923.586,  # throttle^2*Eh
            #     7.647,      # Eh*velocity_x
            #     -0.445,     # velocity_x*exp(-floor_dis)
            #     -0.252      # pitch*exp(-floor_dis)
            # ])
            self.theta_hat = np.zeros(15)

            # data_gain = np.load('Data/controller_gains.npz')
            # self.theta_hat = data_gain['theta_hat']


            self.f_hat = 0.0          # Current disturbance estimate
            self.f_hat_prev = 0.0      # Previous disturbance estimate
            self.f_actual = 0.0        # Actual disturbance
            self.f_actual_smooth = 0.0
            self.f_actual_prev = 0.0   # Previous actual disturbance
            self.gamma = 2.0        # Adaptive gain
            # self.gamma = np.diag([2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0])  # 对新项给予更高学习率
            self.lambda_ = -2.0        # Error feedback gain
            
            self.V = 0.0  # 前向速度（m/s）
            self.V_tip = 120.0  # 叶尖速度估计值（m/s），需根据实际无人机参数调整
            self.V_c = 0.15 * self.V_tip  # 临界速度（Cheeseman建议值）
            
            # Previous states for adaptive observer
            self.prev_throttle = 0.0
            self.prev_pos_z = 0.0
            self.prev_Eh = 0.0

            self.temp_acc = np.zeros(3)   
            self.temp_ang = np.zeros(3)
            
            self.ekf_Q = np.diag([0.002, 0.002])  # 过程噪声协方差
            self.ekf_R = 0.1  # 观测噪声协方差
            self.ekf_P = np.diag([1.0, 1.0])  # 状态估计误差协方差
            self.ekf_x = np.array([0.0, 0.0])  # 状态向量 [f_actual, f_dot]
            
            # 姿态控制器参数 for self.att_vontroller()
            self.s_pos = np.zeros(3)       # auxiliary variable
            self.sum_s_pos = np.zeros(3)
            self.Gamma_pos = 5
            self.K_pos = np.array([2, 2, 10]) * 0.5
            self.Ki_pos = np.array([5, 5, 5]) * 0.2


            # 姿态环 参数
            self.s_att = np.zeros(3)       # auxiliary variable
            self.sum_s_att = np.zeros(3)
            self.sum_att_error = np.zeros(3)
            self.Gamma_att = 3
            self.KP_ATT = np.array([5, 5, 5]) * 0.4
            self.KD_ATT = np.array([2, 2, 2]) * 0.2
            self.KI_ATT = np.array([1, 1, 1]) * 0.2



    ###########################################
    # Select controller by self.controller_flag
    ###########################################
    def get_controller(self, state, state_des, obs_flag):
        '''和上面选择控制参数的名称一样，调用相关的控制器'''
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

        self.pos = self.pos_error + self.pos_des
        self.vel = self.vel_error + self.vel_des
        self.acc = self.acc_error + self.acc_des
        self.att = self.att_error + self.att_des
        self.ang = self.ang_error + self.ang_des

        if self.controller_flag == 'NFC':
            action = self.NFC(obs_flag, state_des)
        elif self.controller_flag == 'NFC_gazebo':
            action = self.NFC_gazebo(obs_flag, state_des)

        elif self.controller_flag == 'MRAC':
            self.ref_input = self.pos_des
            self.model_update()
            action = self.mrac_controller(obs_flag, state_des)  
        elif self.controller_flag == 'RL_MRAC':
            index = round(self.time / self.dt)
            self.pos_ref = self.pos_ref_all[index]
            self.vel_ref = self.vel_ref_all[index]
            self.ref_input = self.ref_input_all[index]
            self.model_update()
            action = self.mrac_controller(obs_flag, state_des)          
        elif self.controller_flag == 'NFC_regressor':
            action = self.NFC_regressor(obs_flag, state_des)

        elif self.controller_flag == 'RL_full_model':
            action = self.RL_full_model(obs_flag)
        elif self.controller_flag == 'RL':
            action = self.RL(obs_flag, state_des)
        elif self.controller_flag == 'RL_gazebo':
            action = self.RL_gazebo(obs_flag, state_des)

        else:
            raise ValueError(
                f"无效的控制器标识: {self.controller_flag}!") 

        return action


    #####################################
    # NFC : for Flare
    #####################################
    def NFC(self, obs_flag, state_des):
        if obs_flag:
            ### Translation control
            self.pos_error = self.pos - self.pos_des
            self.vel_error = self.vel - self.vel_des

            self.s_pos = self.vel_error + self.Gamma_pos * self.pos_error
            self.sum_s_pos = self.sum_s_pos + self.s_pos * self.dt * 10 # 将采样时间从姿态环变成位置环

            self.M = self.mass * np.eye(3)
            self.G = self.mass * np.array([0, 0, self.g])
            
            self.vel_ref = self.vel_des - self.Gamma_pos * self.pos_error
            self.acc_ref = self.acc_des - self.Gamma_pos * self.vel_error

            thrust = np.dot(self.M, self.acc_ref) + self.G - self.K_pos * self.s_pos - self.Ki_pos * self.sum_s_pos

            # Control constrain
            thrust[0] = max(min(thrust[0], self.MAX_ACC * self.mass), -self.MAX_ACC * self.mass)
            thrust[1] = max(min(thrust[1], self.MAX_ACC * self.mass), -self.MAX_ACC * self.mass)
            thrust[2] = max(min(thrust[2], 1.8 * self.MAX_ACC * self.mass), 0.0)

            # pdb.set_trace()
            self.force_controller = thrust

        action = self.force_controller

        # 计算姿态环控制器
        action = self.controller_att.NFC_att(action, self.att, self.ang, state_des)

        return action
    
    #####################################
    # NFC : for Gazebo
    #####################################
    def NFC_gazebo(self, obs_flag, state_des):

        ### Translation control
        self.pos_error = self.pos - self.pos_des
        self.vel_error = self.vel - self.vel_des

        self.s_pos = self.vel_error + self.Gamma_pos * self.pos_error
        self.sum_s_pos = self.sum_s_pos + self.s_pos * self.dt * 10 # 将采样时间从姿态环变成位置环

        self.M = self.mass * np.eye(3)
        self.G = self.mass * np.array([0, 0, self.g])
        
        self.vel_ref = self.vel_des - self.Gamma_pos * self.pos_error
        self.acc_ref = self.acc_des - self.Gamma_pos * self.vel_error

        thrust = np.dot(self.M, self.acc_ref) + self.G - self.K_pos * self.s_pos - self.Ki_pos * self.sum_s_pos

        # Control constrain
        thrust[0] = max(min(thrust[0], self.MAX_ACC * self.mass), -self.MAX_ACC * self.mass)
        thrust[1] = max(min(thrust[1], self.MAX_ACC * self.mass), -self.MAX_ACC * self.mass)
        thrust[2] = max(min(thrust[2], 1.8 * self.MAX_ACC * self.mass), 0.0)

        # pdb.set_trace()
        self.force_controller = thrust

        # 计算姿态环控制器
        thrust_body, att_des = self.controller_att.Decomposition1(thrust, state_des.att)

        # 返回值是：机身推力 + 3维期望姿态
        action = np.array([thrust_body, att_des[0], att_des[1], att_des[2]])

        return action

        



    
    #####################################
    # MRAC + regressor : for Flare
    #####################################
    def mrac_regressor_controller(self, pos_control_flag):

        acc = np.array([self.acc[0], self.acc[1], self.acc[2]])
        self.f_actual = self.mass * acc[2] + self.mass * self.g - self.force_controller[2]
        self.f_actual_smooth = self.update_ekf(self.f_actual)


        xi_prev = self.calculate_library_terms_prev()
        f_tilde = self.f_hat_prev - self.f_actual_prev
        f_dot_hat = np.dot(xi_prev.T, self.theta_hat) + self.lambda_ * f_tilde
        self.f_hat = self.f_hat_prev + f_dot_hat * self.dt
        
        # Update adaptive parameters
        theta_dot = -self.gamma * f_tilde * xi_prev
        self.theta_hat += theta_dot * self.dt

        if pos_control_flag:
            ### Translation control
            self.pos_error = self.pos_des - self.pos
            self.vel_error = self.vel_des - self.vel
            error = np.hstack((self.pos_error, self.vel_error)).reshape(-1,1)
            state = np.hstack((self.pos, self.vel)).reshape(-1,1)
            state_ref = self.rl_input.reshape(-1,1)
            phi = np.hstack((self.pos, self.vel)).reshape(-1,1)

            kx_update = self.gamma_x @ state @ error.T @ self.P @ self.B
            kr_update = self.gamma_r @ state_ref @ error.T @ self.P @ self.B
            theta_update = - self.gamma_theta @ phi @ error.T @ self.P @ self.B
            self.kx = self.kx + kx_update.T * self.dt * 10
            self.kr = self.kr + kr_update.T * self.dt * 10
            self.theta = self.theta + theta_update * self.dt * 10
            self.G = self.mass * np.array([0, 0, self.g]).reshape(-1, 1)
            pos = np.array(self.pos).reshape(-1, 1)
            vel = np.array(self.vel).reshape(-1, 1)

            # thrust = self.kx @ state + self.kr @ state_ref - self.theta.T @ phi + self.G  - self.Am_k*pos - self.Am_k*vel
            thrust = self.kx @ state + self.kr @ state_ref - self.theta.T @ phi + self.G
            
            # Control constrain
            thrust[0] = max(min(thrust[0], self.MAX_ACC * self.mass), -self.MAX_ACC * self.mass)
            thrust[1] = max(min(thrust[1], self.MAX_ACC * self.mass), -self.MAX_ACC * self.mass)
            thrust[2] = max(min(thrust[2], 1.8 * self.MAX_ACC * self.mass), 0.0)

            self.force_controller = thrust.flatten()
            self.Decomposition1()

        
        # 计算 throttle 的大小
        ddx, ddy, ddz = self.force_controller / self.mass

        # Nonlinear feedback controller得到的Fz已经考虑了g了，这里不需要重复考虑
        U1 = self.mass * np.sqrt(ddx ** 2 + ddy ** 2 + (ddz) ** 2)
        self.thrust = U1
        self.throttle = U1 * self.POTT  # 将机体坐标系下的推力转化为油门（0-1）


        self.att_des = np.clip(self.att_decom + self.att_des, -self.MAX_ATT, self.MAX_ATT)
        self.att_acc_des = (self.ang_des - self.temp_ang) / self.dt 
        self.temp_ang = self.ang_des

        self.att_error = self.att - self.att_des
        self.ang_error = self.ang - self.ang_des

        self.s_att = self.ang_error + self.Gamma_att * self.att_error
        self.sum_s_att = self.sum_s_att + self.s_att * self.dt

        self.ang_ref = self.ang_des - self.Gamma_att * self.att_error
        self.att_acc_ref = self.att_acc_des - self.Gamma_att * self.ang_error


        self.sum_att_error = self.sum_att_error + self.att_error * self.dt
        self.torque_controller = np.dot(self.inertial, self.att_acc_ref) + np.cross(self.ang, np.dot(self.inertial, np.array(self.ang))) - self.KP_ATT * self.att_error - self.KD_ATT * self.ang_error - self.KI_ATT * self.sum_att_error

        # Control constrain
        max_torque = 1.0
        self.torque_controller = np.clip(self.torque_controller, -max_torque, max_torque)


        self.tau_roll = self.torque_controller[0]
        self.tau_pitch = self.torque_controller[1]
        self.tau_yaw = self.torque_controller[2]


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
            phi = np.hstack((self.pos, self.vel)).reshape(-1,1)

            kx_update = self.gamma_x @ state @ error.T @ self.P @ self.B
            kr_update = self.gamma_r @ ref_input @ error.T @ self.P @ self.B
            theta_update = - self.gamma_theta @ phi @ error.T @ self.P @ self.B
            self.kx = self.kx + kx_update.T * self.dt * 10
            self.kr = self.kr + kr_update.T * self.dt * 10
            self.theta = self.theta + theta_update * self.dt * 10
            self.G = self.mass * np.array([0, 0, self.g]).reshape(-1, 1)

            # thrust = self.kx @ state + self.kr @ ref_input - self.theta.T @ phi + self.G  - self.Am_k*pos - self.Am_k*vel
            thrust = self.kx @ state + self.kr @ ref_input - self.theta.T @ phi + self.G
            
            # Control constrain
            thrust[0] = max(min(thrust[0], self.MAX_ACC * self.mass), -self.MAX_ACC * self.mass)
            thrust[1] = max(min(thrust[1], self.MAX_ACC * self.mass), -self.MAX_ACC * self.mass)
            thrust[2] = max(min(thrust[2], 1.8 * self.MAX_ACC * self.mass), 0.0)

            self.force_controller = thrust.flatten()

        action = self.force_controller

        # 计算姿态环控制器
        action = self.controller_att.NFC_att(action, self.att, self.ang, state_des)

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
            self.kx = self.kx + kx_update.T * self.dt * 10
            self.kr = self.kr + kr_update.T * self.dt * 10
            self.theta = self.theta + theta_update * self.dt * 10
            self.G = self.mass * np.array([0, 0, self.g]).reshape(-1, 1)
            pos = np.array(self.pos).reshape(-1, 1)
            vel = np.array(self.vel).reshape(-1, 1)

            thrust = self.kx @ state + self.kr @ state_ref - self.theta.T @ phi
            
            # Control constrain
            thrust[0] = max(min(thrust[0], self.MAX_ACC * self.mass), -self.MAX_ACC * self.mass)
            thrust[1] = max(min(thrust[1], self.MAX_ACC * self.mass), -self.MAX_ACC * self.mass)
            thrust[2] = max(min(thrust[2], 1.8 * self.MAX_ACC * self.mass), 0.0)

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
    # RL for Model & Flare
    #####################################
    def RL(self, obs_flag, state_des):
        if obs_flag:
            obs = np.hstack((self.pos_error, self.vel_error, self.att_error, self.ang_error))
            obs_tensor = torch.FloatTensor(obs).to(self.device)
            mean, log_std = self.pi_net(obs_tensor)
            std = log_std.exp()
            
            deterministic=True
            if deterministic:
                action = torch.tanh(mean)
            else:
                z = Normal(0, 1).sample(mean.shape).to(self.device)
                action = torch.tanh(mean + std * z)
            action = action.detach().cpu().numpy()
                
            # RL controller 处理
            scale = np.array([6, 6, 13])    # 13 = 1.32*9.8
            self.force_controller = scale * np.array([action[0], action[1], action[2]+1])


        action = self.force_controller

        # 计算姿态环控制器
        action = self.controller_att.NFC_att(action, self.att, self.ang, state_des)

        return action
    
    #####################################
    # RL for Gazebo
    #####################################
    def RL_gazebo(self, obs_flag, state_des):
        obs = np.hstack((self.pos_error, self.vel_error, self.att_error, self.ang_error))
        obs_tensor = torch.FloatTensor(obs).to(self.device)
        mean, log_std = self.pi_net(obs_tensor)
        std = log_std.exp()

        action = torch.tanh(mean).detach().cpu().numpy()
            
        # RL controller 处理
        scale = np.array([6, 6, 13])    # 13 = 1.32*9.8
        self.force_controller = scale * np.array([action[0], action[1], action[2]+1])

        # 计算姿态环控制器
        thrust_body, att_des = self.controller_att.Decomposition1(self.force_controller, state_des.att)

        # 返回值是：机身推力 + 3维期望姿态
        action = np.array([thrust_body, att_des[0], att_des[1], att_des[2]])

        return action

    
    #####################################
    # RL with full model (4 input, 12 state) for Model
    #####################################
    def RL_full_model(self, obs_flag):
        # if obs_flag:
        obs = 1 * np.hstack((self.pos_error, self.vel_error, self.att_error, self.ang_error))
        obs_tensor = torch.FloatTensor(obs).to(self.device)
        mean, log_std = self.pi_net(obs_tensor)
        std = log_std.exp()
        

        deterministic=True
        if deterministic:
            action = torch.tanh(mean)
        else:
            z = Normal(0, 1).sample(mean.shape).to(self.device)
            action = torch.tanh(mean + std * z)
        action = action.detach().cpu().numpy()
            
        # RL controller 处理
        scale = np.array([13, 1, 1, 1])    # 13 = 1.32*9.8
        action = scale * np.array([action[0], action[1], action[2], action[3]])  # [Thrust, Torque_x, Torque_y, Torque_z]

            # # 是否进行后处理
            # self.G = self.mass * np.array([0, 0, self.g]).reshape(-1, 1)
            # thrust = action_sacle + self.G
            # thrust = action_sacle + self.G - self.Am_k*pos - self.Am_k*vel
            # thrust[0] = max(min(thrust[0], self.MAX_ACC * self.mass), -self.MAX_ACC * self.mass)
            # thrust[1] = max(min(thrust[1], self.MAX_ACC * self.mass), -self.MAX_ACC * self.mass)
            # thrust[2] = max(min(thrust[2], 1.8 * self.MAX_ACC * self.mass), 0.0)

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

    def NFC_regressor(self, obs_flag, state_des):   
        # Always calculate actual disturbance (f_actual)
        self.f_actual = self.mass * self.acc[2] + self.mass * self.g - self.force_controller[2]
        self.f_actual_smooth = self.update_ekf(self.f_actual)


        xi_prev = self.calculate_library_terms_prev()
        f_tilde = self.f_hat_prev - self.f_actual_prev
        f_dot_hat = np.dot(xi_prev.T, self.theta_hat) + self.lambda_ * f_tilde
        self.f_hat = self.f_hat_prev + f_dot_hat * self.dt * 1
        
        # Update adaptive parameters
        theta_dot = -self.gamma * f_tilde * xi_prev
        self.theta_hat += theta_dot * self.dt * 1
        
        # Rest of the controller code remains the same...
        if obs_flag:
            
            ### Translation control
            self.pos_error = self.pos - self.pos_des
            self.vel_error = self.vel - self.vel_des

            self.s_pos = self.vel_error + self.Gamma_pos * self.pos_error
            self.sum_s_pos = self.sum_s_pos + self.s_pos * self.dt * 10 # 将采样时间从姿态环变成位置环

            self.M = self.mass * np.eye(3)
            self.G = self.mass * np.array([0, 0, self.g])
            
            self.vel_ref = self.vel_des - self.Gamma_pos * self.pos_error
            self.acc_ref = self.acc_des - self.Gamma_pos * self.vel_error

            
            thrust = np.dot(self.M, self.acc_ref) + self.G - self.K_pos * self.s_pos - self.Ki_pos * self.sum_s_pos - 1.0 * np.array([0, 0, self.f_hat])

            # Control constrain
            thrust[0] = max(min(thrust[0], self.MAX_ACC * self.mass), -self.MAX_ACC * self.mass)
            thrust[1] = max(min(thrust[1], self.MAX_ACC * self.mass), -self.MAX_ACC * self.mass)
            thrust[2] = max(min(thrust[2], 1.8 * self.MAX_ACC * self.mass), 0.0)

            # pdb.set_trace()
            self.force_controller = thrust

        # Save states for next time step
        self.prev_pos_z = self.pos[2]
        self.f_hat_prev = self.f_hat
        self.f_actual_prev = self.f_actual_smooth
        self.prev_Eh = self.empirical_gef_coe_Sanchez_Cuevas(0.19, self.prev_pos_z, 0.42, 0.6)

        action = self.force_controller

        # 计算姿态环控制器
        action = self.controller_att.NFC_att(action, self.att, self.ang, state_des)
        self.throttle = action[0] * self.POTT
        self.prev_throttle = self.throttle

        return action
        
        
        


    def calculate_library_terms_prev(self):
        
        # 计算极角（弧度，范围(-π, π]）
        object_center = self.pos[0:2]
        theta_rad = np.arctan2(object_center[1], object_center[0])
        
        # 转换为[0, 2π)范围，并转化为degree
        theta_degree = theta_rad % (2 * np.pi) * self.RAD2DEG

        # 判断pos_obs属于第几个障碍物区域(实际上26°差不多，取30.)
        if theta_degree >= 90 and theta_degree < 210:
            floor_center = np.array([-1.5, 0.0])
        elif theta_degree >= 210 and theta_degree < 330:
            floor_center = np.array([0.75, -1.3])
        else:
            floor_center = np.array([0.75, 1.3])

        # 不考虑距离1以内，当作是圆形内部
        floor_dis = np.linalg.norm(object_center - floor_center) - 1.0

        floor_dis = max(floor_dis, 0)
        # 9.75 19 36 64
        
        return np.array([
            self.f_hat_prev * self.prev_throttle * np.exp(-floor_dis),
            self.f_hat_prev * self.prev_pos_z * np.exp(-floor_dis),
            (self.f_hat_prev**2) * self.prev_pos_z * np.exp(-floor_dis),
            self.prev_throttle * self.prev_Eh * np.exp(-floor_dis),
            (self.f_hat_prev**2) * self.prev_Eh * np.exp(-floor_dis),
            (self.prev_throttle**2) * self.prev_Eh * np.exp(-floor_dis),
            self.f_hat_prev * self.prev_throttle,  # velocity_x
            self.f_hat_prev * self.prev_pos_z,     # velocity_x
            (self.f_hat_prev**2) * self.prev_pos_z,
            self.prev_throttle * self.prev_Eh,
            (self.f_hat_prev**2) * self.prev_Eh,
            (self.prev_throttle**2) * self.prev_Eh,
            self.prev_Eh * self.vel[0],
            self.vel[0] * np.exp(-floor_dis),
            self.att[1] * np.exp(-floor_dis),
        ])
    
    def empirical_gef_coe_Sanchez_Cuevas(self, R, z, d, b):
        """计算 Sanchez-Cuevas 系数 E_h"""
        Kb = 0.5
        term1 = 1 - (R / (4 * z)) ** 2 - R ** 2 * (z / np.sqrt((d ** 2 + 4 * z ** 2) ** 3))
        term2 = (R ** 2 / 2) * (z / np.sqrt((2 * d ** 2 + 4 * z ** 2) ** 3))
        term3 = 2 * R ** 2 * (z / np.sqrt((b ** 2 + 4 * z ** 2) ** 3)) * Kb
        return np.round((np.power(term1 - term2 - term3, -1) - 1), 3)
    
    


    def update_ekf(self, z_measurement):
        """
        EKF更新步骤
        z_measurement: 观测值 (f_actual)
        """
        # 预测步骤
        # 状态转移矩阵
        F = np.array([[1, self.dt * 10],
                    [0, 1]])
        
        # 预测状态
        self.ekf_x = F @ self.ekf_x
        
        # 预测协方差
        self.ekf_P = F @ self.ekf_P @ F.T + self.ekf_Q
        
        # 更新步骤
        H = np.array([1, 0])  # 观测矩阵
        y = z_measurement - H @ self.ekf_x  # 残差 y=vk
        S = H @ self.ekf_P @ H.T + self.ekf_R  # 残差协方差
        K = self.ekf_P @ H.T / S  # 卡尔曼增益
        
        # 更新状态和协方差
        self.ekf_x = self.ekf_x + K * y
        self.ekf_P = (np.eye(2) - np.outer(K, H)) @ self.ekf_P
        
        return self.ekf_x[0]  # 返回估计的f_actual
    
    ###################################################
    # 接下来是UAV数学模型的更新函数
    ###################################################
    def model_pos_update(self):
        state = np.hstack((self.pos, self.vel)).reshape(-1,1)
        input = self.force_controller.reshape(-1,1)
        state_update = self.Am @ state + self.B @ input
        next_state = state + state_update * self.dt 
        return next_state.flatten() 
    
    def model_nonlinear_update(self):

        
        current_time = self.time
        next_time = self.time + self.dt

        pos_input = np.array([0, 0, self.force_body])
        att_input = self.torque_controller
        current_pos_state = np.concatenate([self.pos, self.vel], axis=0)
        current_att_state = np.concatenate([self.att, self.ang], axis=0)

        def pos_input_dynamic(t, state):
            return self.pos_dynamic(t, state, pos_input)
        
        def att_input_dynamic(t, state):
            return self.att_dynamic(t, state, att_input)



        sol_pos = solve_ivp(fun=pos_input_dynamic, t_span=[current_time, next_time], y0=current_pos_state, method='RK45', t_eval=[next_time], rtol=1e-6, atol=1e-9)

        sol_att = solve_ivp(fun=att_input_dynamic, t_span=[current_time, next_time], y0=current_att_state, method='RK45', t_eval=[next_time], rtol=1e-6, atol=1e-9)

        return sol_pos, sol_att

    
    def att_dynamic(self, t, state, input):
        # 欧拉角导数
        att_dot = self.euler_derivatives() @ self.ang
        
        # 角加速度 (使用刚体转动方程: I·ω̇ + ω×(I·ω) = M)
        omega = self.ang
        ang_dot = self.J_inv @ (input - np.cross(omega, self.J @ omega))

        att_loop_dot = np.concatenate([att_dot, ang_dot], axis=0)
        return att_loop_dot

    def pos_dynamic(self, t, state, input):

        # 重力向量 (世界坐标系)
        gravity_world = np.array([0, 0, -self.mass * self.g])
        
        # 推力向量 (机体坐标系 -> 世界坐标系)
        R = self.rotation_matrix()
        thrust_world = R @ input
        
        # 计算加速度 (世界坐标系)
        pos_dot = self.vel
        pos_vel_dot = (thrust_world + gravity_world) / self.mass

        pos_loop_dot = np.concatenate([pos_dot, pos_vel_dot], axis=0)
        return pos_loop_dot
    
    def rotation_matrix(self):
        """计算从机体坐标系到世界坐标系的旋转矩阵 (ZYX顺序)"""
        phi, theta, psi = self.att

        # 偏航矩阵 (绕z轴)
        Rz = np.array([
            [np.cos(psi), -np.sin(psi), 0],
            [np.sin(psi), np.cos(psi), 0],
            [0, 0, 1]])
        
        # 俯仰矩阵 (绕y轴)
        Ry = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]])
        
        # 滚转矩阵 (绕x轴)
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi), np.cos(phi)]])
        
        # 总旋转矩阵 R = Rz * Ry * Rx
        return Rz @ Ry @ Rx
    
    def euler_derivatives(self):
        """计算欧拉角导数 (角速度到欧拉角导数的转换)"""
        phi, theta, psi = self.att

        # 转换矩阵
        W = np.array([
            [1, np.sin(phi) * np.tan(theta), np.cos(phi) * np.tan(theta)],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi) / np.cos(theta), np.cos(phi) / np.cos(theta)]
        ])
        
        # 欧拉角导数
        return W
    

    def model_update(self):
        state = np.hstack((self.pos_ref, self.vel_ref)).reshape(-1,1)
        input = self.ref_input.reshape(-1,1)
        state_update = self.Am @ state + self.B @ input
        next_state = state + state_update * self.dt 
        self.pos_ref = next_state.flatten()[:3]
        self.vel_ref = next_state.flatten()[3:]






    

 