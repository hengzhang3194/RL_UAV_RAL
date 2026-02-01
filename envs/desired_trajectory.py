import numpy as np
import math
from scipy import interpolate as interp

from envs.utils import State_struct
import pdb


class Desired_trajectory:
    def __init__(self, trajectory_flag='waypoint', **kwargs):
        '''
        需要设置trajectory_flag参数，来选择使用哪一条轨迹。
        - 'NFC+NFC': position-loop is NFC, attitude-loop is Nonlinear Feedback Controller.
        - 'MRAC+NFC': position-loop is MRAC, attitude-loop is Nonlinear Feedback Controller.
        - 'RL+NFC': position-loop is RL, attitude-loop is Nonlinear Feedback Controller.
        - 'RL-DR+NFC': position-loop is RL with domain randomization, attitude-loop is Nonlinear Feedback Controller.
        - 'MRAC_full_model': full model is MRAC.
        '''

        self.state = State_struct()


        self.trajectory_flag = trajectory_flag
        self.DEG2RAD = math.pi / 180  # 0.01745
        self.RAD2DEG = 180 / math.pi     # 57.2958

        # 是否传入外界的轨迹数据
        if "rl_data" in kwargs:
            self.rl_data = kwargs["rl_data"]

        if self.trajectory_flag == 'bezier':
            self.generate_bezier()
            



    #############################################
    # 获取期望轨迹的随时间的演化      #
    #############################################
    def get_desired_trajectory(self, t):
        '''
        t: current time.
        '''

        if self.trajectory_flag == 'waypoint':
            if t < 20:
                self.state.pos = np.array([0.0, 0.0, 1.2])
            else:
                self.state.pos = np.array([-1.0, 0.0, 1.2])

            self.state.vel = np.array([0.0, 0.0, 0.0])
            self.state.acc = np.array([0.0, 0.0, 0.0])
            self.state.att = np.array([0.0, 0.0, 0.0]) * self.DEG2RAD
            self.state.ang = np.array([0.0, 0.0, 0.0])
            
        elif self.trajectory_flag == 2:
            # rl只训练了位置环的时候
            idx = self.rl_data_index
            self.pos_des = self.rl_data["pos"][idx]
            self.vel_des = self.rl_data["vel"][idx]
            self.att_des = np.array([0.0, 0.0, 0.0]) * self.DEG2RAD
            self.att_vel_des = np.array([0.0, 0.0, 0.0])
            self.rl_input = self.rl_data["rl_input"][idx]
            self.rl_data_index += 1

        elif self.trajectory_flag == 3:
            idx = self.seq  # 获取当前仿真运行的index
            self.pos_des = self.rl_data["pos"][idx]
            self.vel_des = self.rl_data["vel"][idx]
            self.att_des = self.rl_data["att"][idx]
            self.att_vel_des = self.rl_data["att_vel"][idx]
            self.rl_input = self.rl_data["rl_input"][idx]
            print(f"att is {self.att_des}.")

        elif self.trajectory_flag == 4:
            idx = self.rl_data_index  # 获取当前仿真运行的index
            if pos_control_flag:
                self.rl_data_index += 1
            self.pos_des = self.rl_data["pos_des"][idx]
            self.vel_des = self.rl_data["vel_des"][idx]
            self.att_des = self.rl_data["att_des"][idx]
            self.att_vel_des = self.rl_data["att_vel_des"][idx]
            self.rl_input = self.rl_data["rl_input"][idx]

        

        elif self.trajectory_flag == 'horizon_circle':
            hovering_time = 6
            start_time = 10
            pos_z = 1.12
            radius = 1.5  # 圆形轨迹半径（m）
            omega = 0.5  # 角速度（rad/s），控制转圈速度（2π/ω 为周期）
            if t < hovering_time:
                self.state.pos = np.array([0.0, 0.0, pos_z])
                self.state.vel = np.array([0.0, 0.0, 0.0])
                self.state.acc = np.array([0.0, 0.0, 0.0])
            elif t < start_time:
                self.state.pos = np.array([radius, 0.0, pos_z])
                self.state.vel = np.array([0.0, 0.0, 0.0])
                self.state.acc = np.array([0.0, 0.0, 0.0])
            else:
                theta = omega * (t - start_time)  # 角度随时间变化：θ = ω·t'

                self.state.pos = np.array([
                    radius * np.cos(theta),     # x = r·cosθ
                    radius * np.sin(theta),     # y = r·sinθ
                    pos_z                       # z 保持2.0m
                ])
                # pdb.set_trace()

                self.state.vel = np.array([
                    -radius * omega * np.sin(theta),  # vx = -rω·sinθ
                    radius * omega * np.cos(theta),   # vy = rω·cosθ
                    0.0                               # vz = 0
                ])

                self.state.acc = np.array([
                    -radius * (omega **2) * np.cos(theta),  # ax = -rω²·cosθ
                    -radius * (omega** 2) * np.sin(theta),  # ay = -rω²·sinθ
                    0.0                                               # az = 0
                ])

            self.state.att = np.array([0.0, 0.0, 0.0]) 
            self.state.ang = np.array([0.0, 0.0, 0.0])



        elif self.trajectory_flag == 11:
            if t < 10:  # 加速阶段 (0.5 m/s², 2 s → 1 m/s)
                x_acc = 0.2                 # a = 0.5 m/s²
                x_vel = x_acc * (t-5)             # v = a * t
                x_pos = 0.5 * x_acc * (t-5) ** 2  # s = 0.5 * a * t²
                
                
                self.pos_des = np.array([x_pos, 0.0, 0.32])  # 固定高度 0.26 m
                self.vel_des = np.array([x_vel, 0.0, 0.0])
                self.acc_des = np.array([x_acc, 0.0, 0.0])
            elif t < 100:  # 匀速阶段 (1 m/s, 10 s)
                x_acc = 0.0                   # 加速度归零
                x_vel = 1.0                   # 保持 1 m/s
                x_pos = 2.5 + 1.0 * (t - 10)  # 加速段位移 + 匀速段位移
                
                self.pos_des = np.array([x_pos, 0.0, 0.32])
                self.vel_des = np.array([x_vel, 0.0, 0.0])
                self.acc_des = np.array([x_acc, 0.0, 0.0])
            else:  # 停止阶段 (保持最终状态)
                x_pos = 0.5 * 0.5 * 2 ** 2 + 1 * 10  # 总位移 = 加速段 + 匀速段
                self.pos_des = np.array([x_pos, 0.0, 0.32])
                self.vel_des = np.array([0.0, 0.0, 0.0])  # 速度为 0
                self.acc_des = np.array([0.0, 0.0, 0.0])  # 加速度为 0
            
            # 姿态始终为 0
            self.att_des = np.array([0.0, 0.0, 0.0]) * self.DEG2RAD
            self.att_vel_des = np.array([0.0, 0.0, 0.0])


        elif self.trajectory_flag == 12:
            if t < 15:
                self.pos_des = np.array([2.0, 0.0, 4.2])
                self.vel_des = np.array([0.0, 0.0, 0.0])
                self.acc_des = np.array([0.0, 0.0, 0.0])
                self.att_des = np.array([0.0, 0.0, 0.0]) * self.DEG2RAD
                self.att_vel_des = np.array([0.0, 0.0, 0.0])
            else:
                self.radius = 5.0  # 圆形轨迹半径（m）
                self.omega = 0.5  # 角速度（rad/s），控制转圈速度（2π/ω 为周期）
                self.z_height = 4.2  # 水平圆的高度（m）
                t_prime = t - 5  # 从 t=5s 开始计时（t'=0 对应 t=5s）
                theta = self.omega * t_prime  # 角度随时间变化：θ = ω·t'

                self.pos_des = np.array([
                    self.radius * np.cos(theta),  # x = r·cosθ
                    self.radius * np.sin(theta),  # y = r·sinθ
                    self.z_height                  # z 保持2.0m
                ])

                self.vel_des = np.array([
                    -self.radius * self.omega * np.sin(theta),  # vx = -rω·sinθ
                    self.radius * self.omega * np.cos(theta),   # vy = rω·cosθ
                    0.0                                         # vz = 0
                ])

                self.acc_des = np.array([
                    -self.radius * (self.omega **2) * np.cos(theta),  # ax = -rω²·cosθ
                    -self.radius * (self.omega** 2) * np.sin(theta),  # ay = -rω²·sinθ
                    0.0                                               # az = 0
                ])

                self.att_des = np.array([0.0, 0.0, 0.0]) 
                self.att_vel_des = np.array([0.0, 0.0, 0.0])
            self.rl_input = np.array([13.0, 0.0, 0.0, 0.0]).reshape(-1, 1)


        elif self.trajectory_flag == 'bezier':
            # 获取位置 (Position)
            self.pos_des = np.array([self.spline_x(t), self.spline_y(t), self.spline_z(t)])
            
            # 获取速度 (Velocity) - 一阶导
            self.vel_des = np.array([self.spline_x.derivative(1)(t), 
                            self.spline_y.derivative(1)(t), 
                            self.spline_z.derivative(1)(t)])
            
            # 获取加速度 (Acceleration) - 二阶导
            self.acc_des = np.array([self.spline_x.derivative(2)(t), 
                            self.spline_y.derivative(2)(t), 
                            self.spline_z.derivative(2)(t)])
            

            self.att_des = np.array([0.0, 0.0, 0.0]) * self.DEG2RAD
            self.att_vel_des = np.array([0.0, 0.0, 0.0])

        elif self.trajectory_flag == 'horizon_eight':
            period = 20.0   # 周期 [s]
            scale_x = 1.0   # 幅值 [m]
            scale_y = 0.5

            # Lissajous曲线 x = sin(t), y = sin(2t)
            # 调整其频率和振幅以满足速度要求
            omega = 2 * np.pi / period      # 角频率
            px = scale_x * np.sin(omega * t)
            py = scale_y * np.sin(2 * omega * t)
            pz = 1.0

            # 计算速度
            vx = scale_x * omega * np.cos(omega * t)
            vy = scale_y * 2 * omega * np.cos(2 * omega * t)
            vz = 0.0
            ax = -1 * scale_x * omega * omega * np.sin(omega * t)
            ay = -1 * scale_y * 4 * omega * omega * np.sin(2 * omega * t)
            az = 0.0

            self.state.pos = np.array([px, py, pz])
            self.state.vel = np.array([vx, vy, vz])
            self.state.acc = np.array([ax, ay, az])
            self.state.att = np.array([0.0, 0.0, 0.0]) * self.DEG2RAD
            self.state.ang = np.array([0.0, 0.0, 0.0])

        elif self.trajectory_flag == 'horizon_circle_MRAC':
            self.radius = 5.2  # 圆形轨迹半径（m）
            self.omega = 0.5  # 角速度（rad/s），控制转圈速度（2π/ω 为周期）
            self.z_height = 4.5  # 水平圆的高度（m）
            t_prime = t - 5  # 从 t=5s 开始计时（t'=0 对应 t=5s）
            theta = self.omega * t_prime  # 角度随时间变化：θ = ω·t'

            self.state.pos = np.array([
                self.radius * np.cos(theta),  # x = r·cosθ
                self.radius * np.sin(theta),  # y = r·sinθ
                self.z_height                  # z 保持2.0m
            ])

            self.state.vel = np.array([
                -self.radius * self.omega * np.sin(theta),  # vx = -rω·sinθ
                self.radius * self.omega * np.cos(theta),   # vy = rω·cosθ
                0.0                                         # vz = 0
            ])

            self.state.acc = np.array([
                -self.radius * (self.omega **2) * np.cos(theta),  # ax = -rω²·cosθ
                -self.radius * (self.omega** 2) * np.sin(theta),  # ay = -rω²·sinθ
                0.0                                               # az = 0
            ])

            self.state.att = np.array([0.0, 0.0, 0.0]) * self.DEG2RAD
            self.state.ang = np.array([0.0, 0.0, 0.0])

        elif self.trajectory_flag == 'horizon_eight_MRAC':
            period = 20.0   # 周期 [s]
            scale_x = 5.0   # 幅值 [m]
            scale_y = 2.5

            # Lissajous曲线 x = sin(t), y = sin(2t)
            # 调整其频率和振幅以满足速度要求
            omega = 2 * np.pi / period      # 角频率
            px = scale_x * np.sin(omega * t)
            py = scale_y * np.sin(2 * omega * t)
            pz = 4.5

            # 计算速度
            vx = scale_x * omega * np.cos(omega * t)
            vy = scale_y * 2 * omega * np.cos(2 * omega * t)
            vz = 0.0
            ax = -1 * scale_x * omega * omega * np.sin(omega * t)
            ay = -1 * scale_y * 4 * omega * omega * np.sin(2 * omega * t)
            az = 0.0

            self.state.pos = np.array([px, py, pz])
            self.state.vel = np.array([vx, vy, vz])
            self.state.acc = np.array([ax, ay, az])
            self.state.att = np.array([0.0, 0.0, 0.0]) * self.DEG2RAD
            self.state.ang = np.array([0.0, 0.0, 0.0])

        elif self.trajectory_flag == 'spiral':
            z_start = 0.5
            z_rate = 0.02
            radius = 0.75
            n_turns = 2     # 一个周期内有几圈
            period = 10.0   # 周期 [s]

            # 控制角速度，使得 n_turns 圈在 duration 内完成
            omega = 2 * np.pi * n_turns / period  # 10s为一个周期

            # 圆形路径 (x = r*cos(θ), y = r*sin(θ))
            px = radius * np.cos(omega * t)
            py = radius * np.sin(omega * t)
            pz = z_start + z_rate * t

            # 计算速度
            vx = -1 * radius * omega * np.sin(omega * t)
            vy = radius * omega * np.cos(omega * t)
            vz = z_rate
            ax = -1 * radius * omega * omega * np.cos(omega * t)
            ay = -1 * radius * omega * omega * np.sin(omega * t)
            az = 0.0

            self.state.pos = np.array([px, py, pz])
            self.state.vel = np.array([vx, vy, vz])
            self.state.acc = np.array([ax, ay, az])
            self.state.att = np.array([0.0, 0.0, 0.0]) * self.DEG2RAD
            self.state.ang = np.array([0.0, 0.0, 0.0])

        elif self.trajectory_flag == 'smooth_curve':
            '''
            给定一个z轴的起点和终点，创建一个随指数衰减的起飞/landing曲线。
            '''
            z_start = 0.4
            z_end = 1.0
            z_rate = 0.8     # how fast to land

            az = np.exp(-z_rate * t) * (z_rate ** 3 * t - z_rate ** 2) * (
                        z_start - z_end)
            vz = np.exp(-z_rate * t) * (-z_rate ** 2 * t) * (z_start - z_end)
            pz = np.exp(-z_rate * t) * (1 + z_rate * t) * (
                        z_start - z_end) + z_end

            px, py = 0.0, 0.0
            vx, vy = 0.0, 0.0
            ax, ay = 0.0, 0.0
            self.state.pos = np.array([px, py, pz])
            self.state.vel = np.array([vx, vy, vz])
            self.state.acc = np.array([ax, ay, az])
            self.state.att = np.array([0.0, 0.0, 0.0]) * self.DEG2RAD
            self.state.ang = np.array([0.0, 0.0, 0.0])

        elif self.trajectory_flag == 'x_line':
            '''
            给定一个z轴的起点和终点，创建一个随指数衰减的起飞/landing曲线。
            '''
            x_start = 0.0

            ax = 0.0
            vx = 0.5
            px = x_start + vx * t

            pz, py = 1.0, 0.0
            vz, vy = 0.0, 0.0
            az, ay = 0.0, 0.0
            self.state.pos = np.array([px, py, pz])
            self.state.vel = np.array([vx, vy, vz])
            self.state.acc = np.array([ax, ay, az])
            self.state.att = np.array([0.0, 0.0, 0.0]) * self.DEG2RAD
            self.state.ang = np.array([0.0, 0.0, 0.0])
















        return self.state

        
    def generate_bezier(self):
        key_point_count = 6 # 关键点的数量
        duration = 70   # 仿真时间 [s]
        key_t = np.linspace(0, duration, key_point_count)

        # 随机生成关键点
        key_x = np.insert(np.cumsum(np.random.uniform(-1, 1, size=key_point_count-1)), 0, 0.0)
        key_y = np.insert(np.cumsum(np.random.uniform(-1, 1, size=key_point_count-1)), 0, 0.0)
        key_z = np.insert(np.cumsum(np.random.uniform(-0.5, 0.5, size=key_point_count-1)), 0, 0.0)

        # 拟合 B-Spline (保存为成员变量)
        # k=5 保证了二阶导（加速度）也是连续的，适合无人机动力学
        bc = ([(1, 0.0), (2, 0.0)], [(1, 0.0), (2, 0.0)])
        self.spline_x = interp.make_interp_spline(key_t, key_x, k=5, bc_type=bc)
        self.spline_y = interp.make_interp_spline(key_t, key_y, k=5, bc_type=bc)
        self.spline_z = interp.make_interp_spline(key_t, key_z, k=5, bc_type=bc)

    def calculate_att(self, acc, t):
        '''
        TODO
        '''
        ax, ay, az = acc
        # 得到平移信息之后再处理得倒全动力学的期望轨迹
        # 姿态角估算（假设 yaw = 0）
        g = 9.81
        thrust = np.sqrt(ax**2 + ay**2 + (az+g)**2)
        roll = np.arcsin(-ay / thrust)

        # if np.abs(az+g) < 1e-6:
        #     pitch = np.sign(ax) * np.pi / 2
        # else:
        pitch = np.arctan(ax, az+g)

        roll = np.zeros_like(t)  
        pitch = np.zeros_like(t)  
        yaw = np.zeros_like(t)      # psi


        # 姿态角速度（简单导数）
        control_dt = 0.05
        roll_rate = np.gradient(roll, control_dt)
        pitch_rate = np.gradient(pitch, control_dt)
        yaw_rate = np.gradient(yaw, control_dt)

        att = np.array([roll, pitch, yaw])
        ang = np.array([roll_rate, pitch_rate, yaw_rate])

        return att, ang