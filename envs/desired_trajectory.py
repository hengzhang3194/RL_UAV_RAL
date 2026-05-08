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

        # # 是否传入外界的轨迹数据
        # if "rl_data" in kwargs:
        #     self.rl_data = kwargs["rl_data"]

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
                self.state.pos = np.array([0.0, 0.0, 1.0])
            else:
                self.state.pos = np.array([-1.0, 0.0, 1.0])

            self.state.vel = np.array([0.0, 0.0, 0.0])
            self.state.acc = np.array([0.0, 0.0, 0.0])
            self.state.att = np.array([0.0, 0.0, 0.0]) * self.DEG2RAD
            self.state.ang = np.array([0.0, 0.0, 0.0])

        elif self.trajectory_flag == 'bezier':
            pos_z = 1.0

            if not hasattr(self, 'start_time'):
                self.start_time = t
            
            # 获取该轨迹的相对时间
            tau = t - self.start_time

            # 获取位置，一阶导，二阶导
            self.state.pos = np.array([self.spline_x(tau), 
                                       self.spline_y(tau), 
                                       self.spline_z(tau) + 1.0])
            self.state.vel = np.array([self.spline_x.derivative(1)(tau), 
                            self.spline_y.derivative(1)(tau), 
                            self.spline_z.derivative(1)(tau)])
            self.state.acc = np.array([self.spline_x.derivative(2)(tau), 
                            self.spline_y.derivative(2)(tau), 
                            self.spline_z.derivative(2)(tau)])

            self.state.att = np.array([0.0, 0.0, 0.0]) * self.DEG2RAD
            self.state.ang = np.array([0.0, 0.0, 0.0])
            

        elif self.trajectory_flag == 'horizon_circle':
            '''
            如果是直接给MRAC运行, 则 radius = 5.2, pos_z = 4.5  
            '''
            pos_z = 1.0
            period = 20.0      # 周期 [s]
            omega = 2 * np.pi / period    # 角速度
            radius = 1.5  # 圆形轨迹半径（m）
            point_time = 10 # 到达指定点的时间 [s]
            if not hasattr(self, 'start_time'):
                self.start_time = t
            
            # 获取该轨迹的相对时间
            tau = t - self.start_time

            # 分阶段平滑进行轨迹
            if tau < point_time:
                start = 0.0
                end = radius
                duration = point_time     # 规划的执行时间
                px, vx, ax = self.goto_point(tau, start, end, duration)
                py, vy, ay = 0.0, 0.0, 0.0
                pz, vz, az = pos_z, 0.0, 0.0
            
            else:
                theta = omega * (tau - point_time)  # 角度值：θ = ω·t

                px = radius * np.cos(theta)     # x = r·cosθ
                py = radius * np.sin(theta)     # y = r·sinθ
                pz = pos_z                       # z 保持设定高度
               
                vx = -radius * omega * np.sin(theta)  # vx = -rω·sinθ
                vy = radius * omega * np.cos(theta)   # vy = rω·cosθ
                vz = 0.0                               # vz = 0

                ax = -radius * (omega **2) * np.cos(theta)  
                ay = -radius * (omega** 2) * np.sin(theta) 
                az = 0.0       # az = 0

            self.state.pos = np.array([px, py, pz])
            self.state.vel = np.array([vx, vy, vz])
            self.state.acc = np.array([ax, ay, az])
            self.state.att = np.array([0.0, 0.0, 0.0]) 
            self.state.ang = np.array([0.0, 0.0, 0.0])


        elif self.trajectory_flag == 'horizon_eight':
            '''
            如果是直接给MRAC运行, 则 scale_x = 5.0, scale_y = 2.5  
            '''
            period = 20.0   # 周期 [s]
            scale_x = 1.0   # 幅值 [m]
            scale_y = 0.5

            if not hasattr(self, 'start_time'):
                self.start_time = t
            
            # 获取该轨迹的相对时间
            tau = t - self.start_time

            # Lissajous曲线 x = sin(t), y = sin(2t)
            # 调整其频率和振幅以满足速度要求
            omega = 2 * np.pi / period      # 角频率
            px = scale_x * np.sin(omega * tau)
            py = scale_y * np.sin(2 * omega * tau)
            pz = 1.0

            # 计算速度
            vx = scale_x * omega * np.cos(omega * tau)
            vy = scale_y * 2 * omega * np.cos(2 * omega * tau)
            vz = 0.0
            ax = -1 * scale_x * omega * omega * np.sin(omega * tau)
            ay = -1 * scale_y * 4 * omega * omega * np.sin(2 * omega * tau)
            az = 0.0

            self.state.pos = np.array([px, py, pz])
            self.state.vel = np.array([vx, vy, vz])
            self.state.acc = np.array([ax, ay, az])
            self.state.att = np.array([0.0, 0.0, 0.0]) * self.DEG2RAD
            self.state.ang = np.array([0.0, 0.0, 0.0])


        elif self.trajectory_flag == 'spiral':
            # --- 参数定义 ---
            z_start = 1.0
            z_rate = 0.02        # 垂直上升速度 [m/s]
            r_start = 0.05        # 初始半径 [m]
            r_rate = 0.03         # 半径增长速率 [m/s]
            period = 20.0        # 周期 [s]
            omega = 2 * np.pi / period  # 角频率

            if not hasattr(self, 'start_time'):
                self.start_time = t
            
            # 获取该轨迹的相对时间
            tau = t - self.start_time

            # --- 1. 计算当前的实时半径 ---
            # r(t) = r_start + r_rate * t
            r_t = r_start + r_rate * tau

            # --- 2. 位置 (Position) ---
            px = r_t * np.cos(omega * tau)
            py = r_t * np.sin(omega * tau)
            pz = z_start + z_rate * tau

            # 求导 (r*cos)' = r'*cos - r*omega*sin
            vx = r_rate * np.cos(omega * tau) - r_t * omega * np.sin(omega * tau)
            vy = r_rate * np.sin(omega * tau) + r_t * omega * np.cos(omega * tau)
            vz = z_rate

            ax = -2 * r_rate * omega * np.sin(omega * tau) - r_t * (omega**2) * np.cos(omega * tau)
            ay =  2 * r_rate * omega * np.cos(omega * tau) - r_t * (omega**2) * np.sin(omega * tau)
            az = 0.0

            self.state.pos = np.array([px, py, pz])
            self.state.vel = np.array([vx, vy, vz])
            self.state.acc = np.array([ax, ay, az])
            self.state.att = np.array([0.0, 0.0, 0.0]) * self.DEG2RAD
            self.state.ang = np.array([0.0, 0.0, 0.0])

        elif self.trajectory_flag == 'smooth_landingxxx':
            '''
            给定一个z轴的起点和终点，创建一个随指数衰减的起飞/landing曲线。
            '''
            if not hasattr(self, 'start_time'):
                self.start_time = t
            
            # 获取该轨迹的相对时间
            tau = t - self.start_time

            start = 1.0
            end = 0.2
            rate = 0.8     # 规划的执行时间

            az = np.exp(-rate * tau) * (rate ** 3 * tau - rate ** 2) * (
                        start - end)
            vz = np.exp(-rate * tau) * (-rate ** 2 * tau) * (start - end)
            pz = np.exp(-rate * tau) * (1 + rate * tau) * (
                        start - end) + end

            px, vx, ax = 0.0, 0.0, 0.0
            py, vy, ay = 0.0, 0.0, 0.0
            self.state.pos = np.array([px, py, pz])
            self.state.vel = np.array([vx, vy, vz])
            self.state.acc = np.array([ax, ay, az])
            self.state.att = np.array([0.0, 0.0, 0.0]) * self.DEG2RAD
            self.state.ang = np.array([0.0, 0.0, 0.0])

        elif self.trajectory_flag == 'smooth_curve':
            '''
            五次多项式公式，确保初始和结束的速度平滑
            p(s) = z_start + (z_end - z_start) * (10s^3 - 15s^4 + 6s^5)
            '''
            z_start = 0.0
            z_end = 1.5
            T = 8.0  # 任务执行时间

            if t >= T:
                # 超过时间后，停在终点
                pz, vz, az = z_end, 0.0, 0.0
            else:
                s = t / T   # s 是归一化进度 [0, 1]
                poly_p = 10 * s**3 - 15 * s**4 + 6 * s**5
                poly_v = (30 * s**2 - 60 * s**3 + 30 * s**4) / T
                poly_a = (60 * s - 180 * s**2 + 120 * s**3) / (T**2)

                dist = z_end - z_start
                pz = z_start + dist * poly_p
                vz = dist * poly_v
                az = dist * poly_a

            px, vx, ax = 0.0, 0.0, 0.0
            py, vy, ay = 0.0, 0.0, 0.0
            self.state.pos = np.array([px, py, pz])
            self.state.vel = np.array([vx, vy, vz])
            self.state.acc = np.array([ax, ay, az])
            self.state.att = np.array([0.0, 0.0, 0.0]) * self.DEG2RAD
            self.state.ang = np.array([0.0, 0.0, 0.0])


        elif self.trajectory_flag == 'smooth_landing':
            z_start = 1.5
            z_end = 0.2
            T = 8.0  # 任务执行时间

            if not hasattr(self, 'start_time'):
                self.start_time = t
            
            # 获取该轨迹的相对时间
            tau = t - self.start_time

            if tau >= T:
                # 超过时间后，停在终点
                pz, vz, az = z_end, 0.0, 0.0
            else:
                s = tau / T   # s 是归一化进度 [0, 1]
                poly_p = 10 * s**3 - 15 * s**4 + 6 * s**5
                poly_v = (30 * s**2 - 60 * s**3 + 30 * s**4) / T
                poly_a = (60 * s - 180 * s**2 + 120 * s**3) / (T**2)

                dist = z_end - z_start
                pz = z_start + dist * poly_p
                vz = dist * poly_v
                az = dist * poly_a

            px, vx, ax = 0.0, 0.0, 0.0
            py, vy, ay = 0.0, 0.0, 0.0
            self.state.pos = np.array([px, py, pz])
            self.state.vel = np.array([vx, vy, vz])
            self.state.acc = np.array([ax, ay, az])
            self.state.att = np.array([0.0, 0.0, 0.0]) * self.DEG2RAD
            self.state.ang = np.array([0.0, 0.0, 0.0])



        elif self.trajectory_flag == 'horizon_flower':
            pos_z = 1.0
            period = 20.0   # 周期 [s]
            scale = 1.0     # 轨迹整体缩放比例
            point_time = 10.0 # 到达初始点的时间
            
            if not hasattr(self, 'start_time'):
                self.start_time = t
            
            # 获取该轨迹的相对时间
            tau = t - self.start_time

            # 分阶段平滑进行轨迹
            if tau < point_time:
                start = 0.0
                end = 1.0 * scale
                duration = point_time     # 规划的执行时间
                px, vx, ax = self.goto_point(tau, start, end, duration)
                py, vy, ay = 0.0, 0.0, 0.0
                pz, vz, az = pos_z, 0.0, 0.0         
            else:
                omega = 2 * np.pi / period  # 基准频率
                theta = omega * (tau - point_time)  # 角度值：θ = ω·t
                
                # 五角星参数方程 (本质是两个频率的叠加)
                # R: 大圆半径, r: 小圆半径。对于五角星，通常取 R=5r, 或使用如下简化比例：
                # px = R * ( (1-k)*cos(theta) + l*k*cos((1-k)/k * theta) )
                px = scale * (2/3 * np.cos(theta) + 1/3 * np.cos(4 * theta))
                py = scale * (2/3 * np.sin(theta) - 1/3 * np.sin(4 * theta))
                pz = pos_z

                # 计算速度，求一阶导
                vx = scale * (-2/3 * omega * np.sin(theta) - 4/3 * omega * np.sin(4 * theta))
                vy = scale * (2/3 * omega * np.cos(theta) - 4/3 * omega * np.cos(4 * theta))
                vz = 0.0

                # 计算加速度，求二阶导
                ax = scale * (-2/3 * omega**2 * np.cos(theta) - 16/3 * omega**2 * np.cos(4 * theta))
                ay = scale * (-2/3 * omega**2 * np.sin(theta) + 16/3 * omega**2 * np.sin(4 * theta))
                az = 0.0

            self.state.pos = np.array([px, py, pz])
            self.state.vel = np.array([vx, vy, vz])
            self.state.acc = np.array([ax, ay, az])
            self.state.att = np.array([0.0, 0.0, 0.0]) * self.DEG2RAD
            self.state.ang = np.array([0.0, 0.0, 0.0])

        elif self.trajectory_flag == 'horizon_star':
            # --- 参数设置 ---
            period = 30.0      # 完成一圈的总周期 [s]
            radius = 1.5      # 五角星中心到顶点的距离 [m]
            pos_z = 1.0    # 飞行高度 [m]
            point_time = 10.0

            if not hasattr(self, 'start_time'):
                self.start_time = t
            
            # 获取相对时间并进行周期性取模
            tau = t - self.start_time

            # 分阶段平滑进行轨迹
            if tau < point_time:
                start = 0.0
                end = radius
                duration = point_time     # 规划的执行时间
                py, vy, ay = self.goto_point(tau, start, end, duration)
                px, vx, ax = 0.0, 0.0, 0.0
                pz, vz, az = pos_z, 0.0, 0.0

            else: 
                tau = (t - self.start_time - point_time) % period
                # --- 1. 定义五角星的5个顶点 ---
                # 五角星连线顺序：0 -> 144° -> 288° -> 72° -> 216° -> 返回起点
                # 我们预计算5个顶点的坐标
                angles = np.deg2rad([90, 90 + 144, 90 + 288, 90 + 72, 90 + 216, 90])
                v_x = radius * np.cos(angles)
                v_y = radius * np.sin(angles)
                
                # --- 2. 确定当前处于哪一段 ---
                num_segments = 5
                seg_duration = period / num_segments  # 每段直线的时间 (4s)
                seg_idx = int(tau // seg_duration)    # 当前段索引
                seg_idx = min(seg_idx, num_segments - 1) # 防止索引溢出
                
                # 本段的起始点和终点
                p_start = np.array([v_x[seg_idx], v_y[seg_idx], pos_z])
                p_end = np.array([v_x[seg_idx+1], v_y[seg_idx+1], pos_z])
                
                # --- 3. 线性插值计算位置和速度 ---
                ratio = (tau % seg_duration) / seg_duration
                px, py, pz = p_start + (p_end - p_start) * ratio
                vx, vy, vz = (p_end - p_start) / seg_duration
                ax, ay, az = np.array([0.0, 0.0, 0.0])

            self.state.pos = np.array([px, py, pz])
            self.state.vel = np.array([vx, vy, vz])
            self.state.acc = np.array([ax, ay, az])
            self.state.att = np.array([0.0, 0.0, 0.0])
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

    def goto_point(self, time, start, end, duration):
        '''
        --- 返回5次插值的轨迹。
        time: 当前时间
        start: 设定的起点
        end: 设定的终点
        duration: 任务执行时长
        '''
        if time >= duration:
            # 超过时间后，停在终点
            pos, vel, acc = end, 0.0, 0.0
        else:
            
            # p(s) = start + (end - start) * (10s^3 - 15s^4 + 6s^5)
            s = time / duration   # s 是归一化进度 [0, 1]
            poly_p = 10 * s**3 - 15 * s**4 + 6 * s**5
            poly_v = (30 * s**2 - 60 * s**3 + 30 * s**4) / duration
            poly_a = (60 * s - 180 * s**2 + 120 * s**3) / (duration**2)

            dist = end - start
            pos = start + dist * poly_p
            vel = dist * poly_v
            acc = dist * poly_a
        return pos, vel, acc

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