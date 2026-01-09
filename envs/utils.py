import numpy as np
import pandas as pd
import math

class State_struct:
    def __init__(self, pos=np.zeros(3), 
                     vel=np.zeros(3),
                     acc = np.zeros(3),
                     jerk = np.zeros(3), 
                     snap = np.zeros(3),
                     att=np.zeros(3), 
                     ang=np.zeros(3), 
                     time=0.0):
    
        self.pos = pos # R^3
        self.vel = vel # R^3
        self.acc = acc
        self.jerk = jerk
        self.snap = snap
        self.att = att # 欧拉角，rad，R^3
        self.ang = ang # 角速度， rad/s，R^3
        self.time = time    # current time

    def update(self, state):
        self.pos = state.pos
        self.vel = state.vel
        self.acc = state.acc
        self.jerk = state.jerk
        self.snap = state.snap
        self.att = state.att
        self.ang = state.ang
        self.time = state.time

    def get_error(self, state1, state2):
        self.pos = state1.pos - state2.pos
        self.vel = state1.vel - state2.vel
        self.acc = state1.acc - state2.acc
        self.jerk = state1.jerk - state2.jerk
        self.snap = state1.snap - state2.snap
        self.att = state1.att - state2.att
        self.ang = state1.ang - state2.ang
        self.time = state1.time


def euler_to_quaternion(RPY):
    """ 
    将欧拉角（姿态角, rad）转换为四元数。
    """
    roll, pitch, yaw = RPY

    # 计算半角
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    # 计算四元数
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return qx, qy, qz, qw
    

def quaternion_to_euler(quat):
    w, x, y, z = quat
    # Roll (X-axis rotation)
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))

    # Pitch (Y-axis rotation)
    sin_pitch = 2 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sin_pitch, -1.0, 1.0))  # Clip to avoid invalid input due to precision errors

    # Yaw (Z-axis rotation)
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))

    return roll, pitch, yaw

def quaternion_to_rotation_matrix(q):
    """ 将四元数（姿态角）转换为旋转矩阵。
    参数:
    q = (qx, qy, qz, qw) -- 四元数
    返回:
    R -- 对应的旋转矩阵
    """
    qx, qy, qz, qw = q
    R = np.array([
        [1 - 2 * qy ** 2 - 2 * qz ** 2, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
        [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx ** 2 - 2 * qz ** 2, 2 * qy * qz - 2 * qx * qw],
        [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx ** 2 - 2 * qy ** 2]
    ])
    return R


def skew(x):
    """ 返回S()。
    skew-symmetric mapping。
    """
    return np.array([[0, -x[2], x[1]],
                    [x[2], 0, -x[0]],
                    [-x[1], x[0], 0]])


def inv_skew(x):
    """ vee map，返回inverse operation of S()。
    skew-symmetric mapping 的逆操作。
    """
    return np.array([x[2,1], x[0, 2], x[1, 0]]).T


def load_csv_data(file_path='data/rl_data.csv'):

    df = pd.read_csv(file_path)

    # 创建 log 字典并重组数组形式
    log = {
        "pos": df[["pos_x", "pos_y", "pos_z"]].to_numpy(),
        "vel": df[["vel_x", "vel_y", "vel_z"]].to_numpy(),
        "att": df[["att_x", "att_y", "att_z"]].to_numpy(),
        "ang": df[["ang_x", "ang_y", "ang_z"]].to_numpy(),

        "pos_des": df[["pos_xd", "pos_yd", "pos_zd"]].to_numpy(),
        "vel_des": df[["vel_xd", "vel_yd", "vel_zd"]].to_numpy(),
        "att_des": df[["att_xd", "att_yd", "att_zd"]].to_numpy(),
        "ang_des": df[["ang_xd", "ang_yd", "ang_zd"]].to_numpy(),

        "rl_input": df[["rl_input_x", "rl_input_y", "rl_input_z"]].to_numpy(),
        "torque": df[["torque_x", "torque_y", "torque_z"]].to_numpy()
    }
    return log
  