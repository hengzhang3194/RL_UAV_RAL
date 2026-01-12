import sys

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.ticker import AutoMinorLocator
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import pdb

plt.rcParams['font.family'] = 'Arial'

# 一些plot的设置
# fig, ax = plt.subplots(figsize=(5, 5))
# ax.set_xlabel("X [m]")
# ax.set_ylabel("Y [m]")
# ax.set_xlim(-1.8, 1.8)
# ax.set_ylim(-1.8, 1.8)
# ax.set_aspect(1)
# ax.xaxis.set_minor_locator(AutoMinorLocator(1))
# ax.yaxis.set_minor_locator(AutoMinorLocator(1))
# ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)
# ax.grid(linestyle=':')
# ax.text(-1.6,  1.4, r'$U_{\infty}$ = ' + f'{env.flow_velocity[0]}m/s', color='blue', fontsize=15)
# plt.tight_layout()
# plt.savefig('media/trajectory.jpg', bbox_inches='tight', pad_inches=0.05, dpi=1200)



def load_csv_data(file_path='/Data/RL_flare.csv'):
    '''
    加载csv文件，返回一个字典，包含所有数据。
    自动处理动态维度的 theta_hat 参数。
    '''
    load_path = sys.path[0] + file_path
    df = pd.read_csv(load_path)

    # 1. 自动提取所有以 'theta_' 开头的列名，并按序号排序（确保 theta_0, theta_1 顺序正确）
    theta_cols = sorted([col for col in df.columns if col.startswith('theta_')], key=lambda x: int(x.split('_')[1]))

    # 2. 创建基础 log 字典
    log = {
        "time": df["time"].to_numpy(),

        "pos": df[["pos_x", "pos_y", "pos_z"]].to_numpy(),
        "vel": df[["vel_x", "vel_y", "vel_z"]].to_numpy(),

        "att": df[["roll", "pitch", "yaw"]].to_numpy(),
        "ang": df[["roll_vel", "pitch_vel", "yaw_vel"]].to_numpy(),

        "thrust_z": df["thrust_z"].to_numpy(),

        "pos_des": df[["pos_xd", "pos_yd", "pos_zd"]].to_numpy(),
        "vel_des": df[["vel_xd", "vel_yd", "vel_zd"]].to_numpy(),

        "att_des": df[["roll_d", "pitch_d", "yaw_d"]].to_numpy(),
        "ang_des": df[["roll_vel_d", "pitch_vel_d", "yaw_vel_d"]].to_numpy(),
    }

    # 3. 动态添加 theta_hat 数据 (如果存在)
    if theta_cols:
        # 将所有 theta 列合并为一个 [N, 15] 的矩阵
        log["theta_hat"] = df[theta_cols].to_numpy()
        
    return log

def load_csv_data_MRAC(file_path='/Data/RL_flare.csv'):
    '''
    加载csv文件，返回一个字典，包含所有的数据
    '''
    load_path = sys.path[0] + file_path
    df = pd.read_csv(load_path)

    # 创建 log 字典并重组数组形式
    log = {
        "time": df["time"].to_numpy(),

        "pos": df[["pos_x", "pos_y", "pos_z"]].to_numpy(),
        "vel": df[["vel_x", "vel_y", "vel_z"]].to_numpy(),

        "att": df[["roll", "pitch", "yaw"]].to_numpy(),
        "ang": df[["roll_vel", "pitch_vel", "yaw_vel"]].to_numpy(),

        "thrust_z": df["thrust_z"].to_numpy(),


        "pos_des": df[["pos_xd", "pos_yd", "pos_zd"]].to_numpy(),
        "vel_des": df[["vel_xd", "vel_yd", "vel_zd"]].to_numpy(),

        "att_des": df[["roll_d", "pitch_d", "yaw_d"]].to_numpy(),
        "ang_des": df[["roll_vel_d", "pitch_vel_d", "yaw_vel_d"]].to_numpy(),

        "pos_ref": df[["pos_xr", "pos_yr", "pos_zr"]].to_numpy(),
        "vel_ref": df[["vel_xr", "vel_yr", "vel_zr"]].to_numpy()
    }
    return log



def load_npz_data(file_path='/Data/RL_MRAC_flare.npz'):
    '''
    加载csv文件，返回一个字典，包含所有的数据
    '''
    load_path = sys.path[0] + file_path
    data = np.load(load_path)

    # 恢复成 log 形式
    log = {}
    if 'kx' in data:
        log["kx"] = [kx.reshape(3, 6) for kx in data['kx']]
        log["kr"] = [kr.reshape(3, 3) for kr in data['kr']]
        log["theta"] = [theta.reshape(6, 3) for theta in data['theta']]
    
    log["thrust_pos"] = data['thrust']
    log["pos_error"] = data['pos_error']
    log["vel_error"] = data['vel_error']


    # 示例：检查恢复的数据
    # print(log["time"].shape)         # 应该是时间数组
    # print(log["kx"][0].shape)       # 应该是 (3, 6)
    # print(log["kr"][0].shape)       # 应该是 (3, 3)
    # print(log["theta"][0].shape)    # 应该是 (6, 3)
    # print(log["thrust_pos"].shape)  # 应该是 (N, 3)
    return log


















#############################################
# Plot here                                 #
#############################################
color1 = 'tab:red'
color2 = 'tab:brown'
color3 = 'tab:cyan'
color4 = 'tab:pink'
color5 = 'tab:olive'
color6 = 'tab:purple'


def plot_pos_att(log):
    '''
    plot position and attitude, 6张子图，每张子图包含两个轴，分别为位置和速度
    '''
    fig, axes = plt.subplots(2, 3)
    ax1 = axes[0][0]
    ax1.plot(log['time'], log['pos'][:, 0], label='pos_x', color=color1)
    ax1.plot(log['time'], log['pos_des'][:, 0], label='pos_xd', color=color2)
    # ax1.plot(log['time'], log['pos_ref'][:, 0], label='pos_xr', color=color6)
    ax1.tick_params(axis='y', labelcolor=color1)

    ax11 = ax1.twinx()
    ax11.plot(log['time'], log['vel'][:, 0], label='vel_x', color=color3)
    ax11.plot(log['time'], log['vel_des'][:, 0], label='vel_xd', color=color4)

    # 分别获取 两个轴 的图例句柄和标签, 并进行展示
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('Position (m)', color=color1)
    ax11.set_ylabel('Velocity (m/s)', color=color3)
    ax1.grid('True')
    line_1, label_1 = ax1.get_legend_handles_labels()
    line_2, label_2 = ax11.get_legend_handles_labels()
    ax1.legend(line_1 + line_2, label_1 + label_2, loc='upper left')


    ax2 = axes[0][1]
    ax2.plot(log['time'], log['pos'][:, 1], label='pos_y', color=color1)
    ax2.plot(log['time'], log['pos_des'][:, 1], label='pos_yd', color=color2)
    # ax2.plot(log['time'], log['pos_ref'][:, 1], label='pos_yr', color=color6)
    ax2.tick_params(axis='y', labelcolor=color1)

    ax21 = ax2.twinx()
    ax21.plot(log['time'], log['vel'][:, 1], label='vel_y', color=color3)
    ax21.plot(log['time'], log['vel_des'][:, 1], label='vel_yd', color=color4)

    # 分别获取 两个轴 的图例句柄和标签, 并进行展示
    ax2.set_xlabel('time (s)')
    ax2.set_ylabel('Position (m)', color=color1)
    ax21.set_ylabel('Velocity (m/s)', color=color3)
    ax2.grid('True')
    line_1, label_1 = ax2.get_legend_handles_labels()
    line_2, label_2 = ax21.get_legend_handles_labels()
    ax2.legend(line_1 + line_2, label_1 + label_2, loc='upper left')


    ax3 = axes[0][2]
    ax3.plot(log['time'], log['pos'][:, 2], label='pos_z', color=color1)
    ax3.plot(log['time'], log['pos_des'][:, 2], label='pos_zd', color=color2)
    # ax3.plot(log['time'], log['pos_ref'][:, 2], label='pos_zr', color=color6)
    ax3.tick_params(axis='y', labelcolor=color1)

    ax31 = ax3.twinx()
    ax31.plot(log['time'], log['vel'][:, 2], label='vel_z', color=color3)
    ax31.plot(log['time'], log['vel_des'][:, 2], label='vel_zd', color=color4)

    # 分别获取 两个轴 的图例句柄和标签, 并进行展示
    ax3.set_xlabel('time (s)')
    ax3.set_ylabel('Position (m)', color=color1)
    ax31.set_ylabel('Velocity (m/s)', color=color3)
    ax3.grid('True')
    line_1, label_1 = ax3.get_legend_handles_labels()
    line_2, label_2 = ax31.get_legend_handles_labels()
    ax3.legend(line_1 + line_2, label_1 + label_2, loc='upper left')


    ax1 = axes[1][0]
    ax1.plot(log['time'], log['att'][:, 0], label='att_x', color=color1)
    ax1.plot(log['time'], log['att_des'][:, 0], label='att_xd', color=color2)
    ax1.tick_params(axis='y', labelcolor=color1)

    ax11 = ax1.twinx()
    ax11.plot(log['time'], log['ang'][:, 0], label='ang_x', color=color3)
    ax11.plot(log['time'], log['ang_des'][:, 0], label='ang_xd', color=color4)

    # 分别获取 两个轴 的图例句柄和标签, 并进行展示
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('Angle (rad)', color=color1)
    ax11.set_ylabel('Angle_velocity (rad/s)', color=color3)
    ax1.grid('True')
    line_1, label_1 = ax1.get_legend_handles_labels()
    line_2, label_2 = ax11.get_legend_handles_labels()
    ax1.legend(line_1 + line_2, label_1 + label_2, loc='upper left')


    ax1 = axes[1][1]
    ax1.plot(log['time'], log['att'][:, 1], label='att_y', color=color1)
    ax1.plot(log['time'], log['att_des'][:, 1], label='att_yd', color=color2)
    ax1.tick_params(axis='y', labelcolor=color1)

    ax11 = ax1.twinx()
    ax11.plot(log['time'], log['ang'][:, 1], label='ang_y', color=color3)
    ax11.plot(log['time'], log['ang_des'][:, 1], label='ang_yd', color=color4)

    # 分别获取 两个轴 的图例句柄和标签, 并进行展示
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('Angle (rad)', color=color1)
    ax11.set_ylabel('Angle_velocity (rad/s)', color=color3)
    ax1.grid('True')
    line_1, label_1 = ax1.get_legend_handles_labels()
    line_2, label_2 = ax11.get_legend_handles_labels()
    ax1.legend(line_1 + line_2, label_1 + label_2, loc='upper left')


    ax1 = axes[1][2]
    ax1.plot(log['time'], log['att'][:, 2], label='att_z', color=color1)
    ax1.plot(log['time'], log['att_des'][:, 2], label='att_zd', color=color2)
    ax1.tick_params(axis='y', labelcolor=color1)

    ax11 = ax1.twinx()
    ax11.plot(log['time'], log['ang'][:, 2], label='ang_z', color=color3)
    ax11.plot(log['time'], log['ang_des'][:, 2], label='ang_zd', color=color4)

    # 分别获取 两个轴 的图例句柄和标签, 并进行展示
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('Angle (rad)', color=color1)
    ax11.set_ylabel('Angle_velocity (rad/s)', color=color3)
    ax1.grid('True')
    line_1, label_1 = ax1.get_legend_handles_labels()
    line_2, label_2 = ax11.get_legend_handles_labels()
    ax1.legend(line_1 + line_2, label_1 + label_2, loc='upper left')

    fig.set_size_inches(15.0, 8.0)
    fig.tight_layout()
    plt.show()

def plot_3d_trajectory(pos, vel, pos_des=None):
    """
    绘制三维轨迹，其中颜色表示速度大小，支持添加期望轨迹（虚线）。
    
    参数：
        pos: (N, 3) numpy array，实际轨迹位置
        vel: (N, 3) numpy array，对应速度向量
        pos_des: (N, 3) numpy array，可选，期望轨迹位置（虚线）
    """
    speed = np.linalg.norm(vel, axis=1)
    points = pos.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # 创建颜色映射
    norm = Normalize(vmin=speed.min(), vmax=speed.max())
    cmap = plt.colormaps['viridis']
    colors = cmap(norm(speed[:-1]))

    # 创建 Line3DCollection
    lc = Line3DCollection(segments, colors=colors, linewidths=2)

    # 绘图
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.add_collection3d(lc)

    # 如果有期望轨迹，就绘制虚线
    if pos_des is not None:
        ax.plot(pos_des[:, 0], pos_des[:, 1], pos_des[:, 2],
                linestyle='--', color='black', linewidth=1.5, label='Reference Trajectory')

    # 设置范围
    ax.set_xlim(pos[:, 0].min(), pos[:, 0].max())
    ax.set_ylim(pos[:, 1].min(), pos[:, 1].max())
    ax.set_zlim(pos[:, 2].min(), pos[:, 2].max())

    # 添加颜色条
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(speed)
    fig.colorbar(mappable, ax=ax, label='Speed')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Trajectory with Speed Coloring and Reference Path')
    set_axes_equal(ax)  # 确保坐标轴比例一致

    if pos_des is not None:
        ax.legend()

    plt.tight_layout()
    # plt.show()


def plot_3d_trajectory_with_error(pos, pos_des=None):
    """
    绘制三维轨迹，其中颜色表示当前轨迹与期望轨迹的误差，支持添加期望轨迹（虚线）。

    参数：
        pos: (N, 3) numpy array，实际轨迹位置
        vel: (N, 3) numpy array，对应速度向量（可选，不用于颜色映射）
        pos_des: (N, 3) numpy array，可选，期望轨迹位置（虚线）
    """
    if pos_des is None:
        raise ValueError("要用误差作为颜色，需要提供 pos_des。")

    # 计算误差（欧氏距离）
    error = np.linalg.norm(pos - pos_des, axis=1)

    points = pos.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # 颜色映射
    norm = Normalize(vmin=error.min(), vmax=error.max())
    cmap = plt.colormaps['viridis']
    colors = cmap(norm(error[:-1]))

    # 创建线段
    lc = Line3DCollection(segments, colors=colors, linewidths=2)

    # 绘图
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.add_collection3d(lc)

    # 虚线绘制期望轨迹
    ax.plot(pos_des[:, 0], pos_des[:, 1], pos_des[:, 2],
            linestyle='--', color='black', linewidth=1.5, label='Reference Trajectory')

    ax.set_xlim(pos[:, 0].min(), pos[:, 0].max())
    ax.set_ylim(pos[:, 1].min(), pos[:, 1].max())
    ax.set_zlim(pos[:, 2].min(), pos[:, 2].max())

    # 添加颜色条
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(error)
    fig.colorbar(mappable, ax=ax, label='Position Error')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Trajectory Colored by Position Error')

    set_axes_equal(ax)
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_3d_trajectory_with_error2(pos, pos_des=None):
    """
    绘制三维轨迹，颜色表示实际轨迹到期望轨迹最近邻误差。

    参数：
        pos: (N, 3) numpy array，实际轨迹位置
        pos_des: (M, 3) numpy array，期望轨迹位置
    """
    if pos_des is None:
        raise ValueError("要用误差作为颜色，需要提供 pos_des。")

    # 创建 cKDTree 搜索
    tree = cKDTree(pos_des)
    dists, _ = tree.query(pos, k=1)  # k=1就是最近邻
    error = dists

    # 绘制轨迹
    points = pos.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # 颜色映射
    norm = Normalize(vmin=error.min(), vmax=error.max())
    cmap = plt.colormaps['viridis']
    colors = cmap(norm(error[:-1]))

    # 创建3D线段集合
    lc = Line3DCollection(segments, colors=colors, linewidths=2)

    # 绘图
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.add_collection3d(lc)

    # 绘制期望轨迹
    ax.plot(pos_des[:, 0], pos_des[:, 1], pos_des[:, 2],
            linestyle='--', color='black', linewidth=1.5, label='Reference Trajectory')

    # 设置范围
    all_points = np.vstack((pos, pos_des))
    ax.set_xlim(all_points[:, 0].min(), all_points[:, 0].max())
    ax.set_ylim(all_points[:, 1].min(), all_points[:, 1].max())
    ax.set_zlim(all_points[:, 2].min(), all_points[:, 2].max())

    # 添加颜色条
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(error)
    fig.colorbar(mappable, ax=ax, label='Spatial Error')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Trajectory Colored by Spatial Nearest Error')

    set_axes_equal(ax)
    ax.legend()
    plt.tight_layout()
    # plt.show()


def plot_3d_trajectory_with_nearest_error_and_matches(pos, pos_des, highlight_threshold=0.8, draw_matches=True):
    """
    绘制三维轨迹，颜色表示每个实际轨迹点到期望轨迹的最近邻误差，
    可以可视化实际点和参考点的最近配对。
    """
    if pos_des is None:
        raise ValueError("需要提供 pos_des。")

    # 最近邻查找
    tree = cKDTree(pos_des)
    nearest_dists, nearest_idx = tree.query(pos, k=1)

    points = pos.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # 颜色映射
    vmax = np.percentile(nearest_dists, 90)  # 95%分位数
    norm = Normalize(vmin=nearest_dists.min(), vmax=vmax)
    cmap = plt.colormaps['viridis']
    colors = cmap(norm(nearest_dists[:-1]))

    lc = Line3DCollection(segments, colors=colors, linewidths=2)

    # 绘图
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.add_collection3d(lc)

    # 绘制参考轨迹
    ax.plot(pos_des[:, 0], pos_des[:, 1], pos_des[:, 2],
            linestyle='--', color='black', linewidth=1.5, label='Reference Trajectory')

    # 高亮大误差点
    # threshold_value = highlight_threshold * nearest_dists.max()
    # large_error_idx = np.where(nearest_dists >= threshold_value)[0]
    # if len(large_error_idx) > 0:
    #     ax.scatter(pos[large_error_idx, 0],
    #                pos[large_error_idx, 1],
    #                pos[large_error_idx, 2],
    #                color='red', s=30, label='Large Error Points')

    # 额外绘制匹配连线
    if draw_matches:
        match_segments = []
        for i in range(0, len(pos), 10):  # 每隔10个点连一次，避免太密
            p_real = pos[i]
            p_ref = pos_des[nearest_idx[i]]
            match_segments.append([p_real, p_ref])
        match_lines = Line3DCollection(match_segments, colors='gray', linewidths=0.5, linestyles='dashed')
        ax.add_collection3d(match_lines)

    # 设置范围
    all_points = np.vstack((pos, pos_des))
    ax.set_xlim(all_points[:, 0].min(), all_points[:, 0].max())
    ax.set_ylim(all_points[:, 1].min(), all_points[:, 1].max())
    ax.set_zlim(all_points[:, 2].min(), all_points[:, 2].max())

    # 颜色条
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(nearest_dists)
    fig.colorbar(mappable, ax=ax, label='Spatial Nearest Error')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Trajectory with Nearest Error and Matching Lines')

    set_axes_equal(ax)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_throttle(log):
    '''
    plot throttle and torque, 6张子图，每张子图包含两个轴，分别为位置和速度
    '''
    fig, axes = plt.subplots(1, 2)
    ax7 = axes[0]
    ax7.set_title('Throttle vs. Time')
    ax7.set_xlabel('time(s)')
    ax7.set_ylabel('Throttle (0-1)', color=color1)
    # ax7.plot(log['time'], log['throttle'], label='throttle', color=color2)
    ax7.tick_params(axis='y', labelcolor=color1)

    ax71 = ax7.twinx()
    ax71.plot(log['time'], log['thrust_z'], label='thrust_des', color=color3)

    # 分别获取 两个轴 的图例句柄和标签, 并进行展示
    lines_1, labels_1 = ax7.get_legend_handles_labels()
    lines_2, labels_2 = ax71.get_legend_handles_labels()
    ax7.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    # ax9 = axes[1]
    # ax9.set_title('Torque vs. Time')
    # ax9.set_xlabel('time(s)')
    # ax9.set_ylabel('Torque (Nm)', color=color3)
    # ax9.plot(log['time'], log['torque_controller'][:, 0], label='torque_roll_des', color=color1)
    # ax9.plot(log['time'], log['torque_controller'][:, 1], label='torque_pitch_des', color=color2)
    # ax9.plot(log['time'], log['torque_controller'][:, 2], label='torque_yaw_des', color=color3)
    # ax9.tick_params(axis='y', labelcolor=color1)
    # ax9.legend()

    fig.set_size_inches(18.0, 10.0)
    fig.tight_layout()
    plt.show()


    
def calculate_rms(log):
    # ################################################
    # # 画 MRAC 的 kx, kr, theta 的每个元素随时间变化
    # ################################################
    pos_error = np.array(log['pos_error'])
    vel_error = np.array(log['vel_error'])

    # 以200个点为一圈，计算每段轨迹误差的 RMS
    pos_rms_segments = compute_segmented_rms(pos_error)
    vel_rms_segments = compute_segmented_rms(vel_error)

    # 输出
    for i, (pr, vr) in enumerate(zip(pos_rms_segments, vel_rms_segments)):
        print(f"Segment {i}: Pos RMS = {pr:.4f}, Vel RMS = {vr:.4f}")



def plot_controller_gains(log, dt=0.05):
    kx_history = np.array(log['kx'])
    kr_history = np.array(log['kr'])
    theta_history = np.array(log['theta'])
    T = kx_history.shape[0]
    time = np.arange(T) * dt  # 横轴时间

    # 画 Kx 的每个元素随时间变化（共 3x6 = 18 条线）
    fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for i in range(3):
        for j in range(6):
            ax[i].plot(time, kx_history[:, i, j], label=f"Kx[{i},{j}]")
        ax[i].legend(loc='upper right')
        ax[i].set_ylabel(f"Kx row {i}")
    ax[-1].set_xlabel("Time [s]")
    fig.suptitle("Kx Parameter Evolution Over Time")
    fig.tight_layout()
    # plt.show()

    # 示例：Kr 的绘图
    fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for i in range(3):
        for j in range(3):
            ax[i].plot(time, kr_history[:, i, j], label=f"Kr[{i},{j}]")
        ax[i].legend()
        ax[i].set_ylabel(f"Kr row {i}")
    ax[-1].set_xlabel("Time [s]")
    fig.suptitle("Kr Parameter Evolution Over Time")
    fig.tight_layout()
    # plt.show()

    # 示例：theta 的绘图
    fig, ax = plt.subplots(6, 1, figsize=(12, 12), sharex=True)
    for i in range(6):
        for j in range(3):
            ax[i].plot(time, theta_history[:, i, j], label=f"θ[{i},{j}]")
        ax[i].legend()
        ax[i].set_ylabel(f"θ row {i}")
    ax[-1].set_xlabel("Time [s]")
    fig.suptitle("Theta Parameter Evolution Over Time")
    fig.tight_layout()
    # plt.show()


    # 假设 thrust_history 是一个 N x 3 的列表或数组
    thrust_history = np.array(log['thrust_pos'])  # 转为 numpy 数组
    # 创建图形
    plt.figure(figsize=(10, 6))
    plt.plot(time, thrust_history[:, 0], label='Thrust X', color='r')
    plt.plot(time, thrust_history[:, 1], label='Thrust Y', color='g')
    plt.plot(time, thrust_history[:, 2], label='Thrust Z', color='b')

    # 添加标签和图例
    plt.xlabel('Time Step')
    plt.ylabel('Thrust')
    plt.title('Thrust Time Series (X, Y, Z)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def compute_segmented_rms(signal, segment_length=200):
    """
    将信号按段划分，每段计算 RMS。
    :param signal: shape 为 (N, 3) 的 numpy 数组
    :param segment_length: 每段的长度
    :return: 一个 list，每个元素是对应段的 RMS 值
    """
    num_segments = signal.shape[0] // segment_length
    rms_list = []

    for i in range(num_segments):
        segment = signal[i * segment_length : (i + 1) * segment_length, :]
        rms = np.sqrt(np.mean(np.sum(segment**2, axis=1)))  # 3维 RMS
        rms_list.append(rms)

    return rms_list



def set_axes_equal(ax):
    """设置3D坐标轴等比例缩放（单位长度视觉上等长）"""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    max_range = max(x_range, y_range, z_range)

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    ax.set_xlim3d([x_middle - max_range / 2, x_middle + max_range / 2])
    ax.set_ylim3d([y_middle - max_range / 2, y_middle + max_range / 2])
    ax.set_zlim3d([z_middle - max_range / 2, z_middle + max_range / 2])




if __name__ == '__main__':
    path = '/Data/RL_MRAC'
    # path = '/Data/RL_MRAC_model'
    # path = '/Data/RL_flare'
    # path = '/Data/RL_model'
    # path = '/Data/RL_random_flare'
    # path = '/Data/RL_random_model'

    
    log = load_csv_data(path+'.csv')
    # log = load_csv_data_MRAC(path+'.csv')
    plot_pos_att(log)

    # fig = plt.figure(figsize=(12, 8))
    # # 绘制参考轨迹
    # plt.plot(log['time'], log['theta_hat_1'],
    #         linestyle='--', color='black', linewidth=1.5, label='Reference 0')
    # plt.plot(log['time'], log['theta_hat_2'],
    #         linestyle='--', color='red', linewidth=1.5, label='Reference 1')
    # plt.show()

    # 如果想看第 5 个参数的随时间变化
    if "theta_hat" in log:
        plt.plot(log["time"], log["theta_hat"][:, 4]) 
        
        # 或者一次性画出所有自适应参数的收敛情况
        plt.plot(log["time"], log["theta_hat"])



    # plot_throttle(log)
    # plot_3d_trajectory(log['pos'], log['vel'], log["pos_des"])
    plot_3d_trajectory_with_error2(log['pos'], log["pos_des"])
    plot_3d_trajectory_with_nearest_error_and_matches(log['pos'], log["pos_des"])


    # log2 = load_npz_data(path+'.npz')
    # calculate_rms(log2)
    # plot_controller_gains(log2)

    plt.show()







