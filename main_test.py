
import sys
import os
sys.path.append('../')
# sys.path.insert(0, 'e:\\Zhangheng\\DATT_v2\\DATT_new')

import numpy as np
import pandas as pd
import time
import pdb

from collections import defaultdict
# from envs.env_gazebo import DroneEnv as DroneEnv_gazebo
from envs.env_flare import DroneEnv as DroneEnv_flare
from envs.env_model import DroneEnv as DroneEnv_model
from envs.env_nonlinear_model import DroneEnv as DroneEnv_nomodel
from envs.desired_trajectory import Desired_trajectory
from envs.controller import Controller
# from envs.env_gazebo import DroneEnv


drone = DroneEnv_flare()
controller1 = Controller(controller_flag='RL_MRAC')
controller2 = Controller(controller_flag='RL_MRAC')
desired_trajectory1 = Desired_trajectory(trajectory_flag='smooth_curve')
desired_trajectory2 = Desired_trajectory(trajectory_flag='horizon_eight')

log = defaultdict(list)  # 用于存储信息的字典

np.set_printoptions(precision=3, suppress=True, floatmode='fixed', linewidth=150)  # 全局调整 NumPy 数组的打印格式

#########################################################
# Simulation loop (simulation time, not physical time)
#########################################################
start_timestamp = time.perf_counter()
_, obs = drone.reset()
obs_flag = False
last_export_time = time.time()
while (drone.state.time <= drone.duration):
    export_time = time.time()

    if drone.state.time < 10.0:
        desired_trajectory = desired_trajectory1
        controller = controller2
    else:
        desired_trajectory = desired_trajectory2
        controller = controller2


    # 获取当前时刻的期望轨迹的信息
    state_des = desired_trajectory.get_desired_trajectory(drone.state.time)

    print(f'Exporting {drone.seq}th ({drone.state.time:.2f}/{drone.duration}s complete), time ratio: {drone.dt / ((export_time - last_export_time) + 1e-10) * 100:.2f}%')

    action = controller.get_controller(obs, state_des, obs_flag)
    print(f"Action is {action}.")



    # if pos_control_flag:
    #     log['kx'].append(drone.kx)
    #     log['kr'].append(drone.kr)
    #     log['theta'].append(drone.theta)
    #     log['thrust_pos'].append(drone.force_controller)
    #     log['pos_error'].append(drone.pos_error)
    #     log['vel_error'].append(drone.vel_error)

    # step simulator
    obs, reward, terminated, truncated, info = drone.step(action, state_des)
    obs = info["obs"]
    obs_flag = info["obs_flag"]
    last_export_time = export_time
    # pdb.set_trace()
    print(f"Obs is {obs.pos}.")


    #################################################
    # Print sensor data and fill plot sequence here #
    #################################################
    # 只有flag为True的时候才记录数据，否则不记录。
    if drone.log_flag:
        log["time"].append(drone.state.time)
        log["throttle"].append(controller.throttle)
        log["thrust"].append(controller.thrust)    # 机体坐标系下的Z-axis推力

        log["pos"].append(drone.state.pos)
        log["vel"].append(drone.state.vel)
        log["att"].append(drone.state.att)
        log["ang"].append(drone.state.ang)
        log["pos_obs"].append(obs.pos)
        log["vel_obs"].append(obs.vel)
        log["pos_des"].append(state_des.pos)
        log["vel_des"].append(state_des.vel)
        log["att_des"].append(state_des.att)
        log["ang_des"].append(state_des.ang)
        log["torque_controller"].append(controller.torque_controller)
        log["force_controller"].append(controller.force_controller)
        

        if controller.controller_flag == 'MRAC':
            log["pos_ref"].append(controller.pos_ref)
            log["vel_ref"].append(controller.vel_ref)

        if 'regressor' in controller.controller_flag:
            log["theta_hat"].append(controller.theta_hat.copy())


    # print(f'Position is: ({drone.pos[0]:.2f}, {drone.pos[1]:.2f}, {drone.pos[2]:.2f}), Attitude is: ({drone.att[0] * drone.RAD2DEG:.3f}°, {drone.att[1] * drone.RAD2DEG:.3f}°, {drone.att[2] * drone.RAD2DEG:.3f}°).')

drone.close()

print(f"Total Cost Real Time: {round(time.perf_counter() - start_timestamp, 4)}s.")
# print(f"Finally, kx is {drone.kx}, kr is {drone.kr}, theta is {drone.theta}.")
# np.savez('Data/controller_gains.npz', kx=controller.kx, kr=controller.kr, theta=controller.theta)
# np.savez('Data/controller_gains.npz', theta_hat=controller.theta_hat)



####################################
# 保存为 CSV 文件
####################################

save_path = sys.path[0] + '/Data/RL_MRAC.csv'

# 2. 将 list 转换为 numpy array (保持你原有的预处理)
for key in log:
    log[key] = np.array(log[key])

# 3. 使用字典构造 DataFrame (Pandas 方式)
# 这种方式直接通过切片将 [N, 3] 的矩阵拆解为 x, y, z 列
storage_dict = {
    'time': log["time"],
    'pos_x': log["pos"][:, 0], 
    'pos_y': log["pos"][:, 1], 
    'pos_z': log["pos"][:, 2],
    'vel_x': log["vel"][:, 0], 
    'vel_y': log["vel"][:, 1], 
    'vel_z': log["vel"][:, 2],
    'pos_xo': log["pos_obs"][:, 0], 
    'pos_yo': log["pos_obs"][:, 1], 
    'pos_zo': log["pos_obs"][:, 2],
    'vel_xo': log["vel_obs"][:, 0],
    'vel_yo': log["vel_obs"][:, 1], 
    'vel_zo': log["vel_obs"][:, 2],
    'roll':  log["att"][:, 0], 
    'pitch': log["att"][:, 1], 
    'yaw':   log["att"][:, 2],
    'roll_vel': log["ang"][:, 0], 
    'pitch_vel': log["ang"][:, 1], 
    'yaw_vel': log["ang"][:, 2],
    # 控制量
    'thrust_z': log["thrust"],  # 假设 thrust 是 1 维的
    'force_x': log["force_controller"][:, 0], 
    'force_y': log["force_controller"][:, 1], 
    'force_z': log["force_controller"][:, 2],
    'torque_x': log["torque_controller"][:, 0], 
    'torque_y': log["torque_controller"][:, 1], 
    'torque_z': log["torque_controller"][:, 2],
    # 期望值
    'pos_xd': log["pos_des"][:, 0], 
    'pos_yd': log["pos_des"][:, 1], 
    'pos_zd': log["pos_des"][:, 2],
    'vel_xd': log["vel_des"][:, 0], 
    'vel_yd': log["vel_des"][:, 1], 
    'vel_zd': log["vel_des"][:, 2],
    'roll_d': log["att_des"][:, 0], 
    'pitch_d': log["att_des"][:, 1], 
    'yaw_d': log["att_des"][:, 2],
    'roll_vel_d': log["ang_des"][:, 0], 
    'pitch_vel_d': log["ang_des"][:, 1], 
    'yaw_vel_d': log["ang_des"][:, 2],
}

# 4. 处理条件分支 (MRAC)
if controller.controller_flag == 'MRAC':
    storage_dict.update({
        'pos_xr': log["pos_ref"][:, 0], 'pos_yr': log["pos_ref"][:, 1], 'pos_zr': log["pos_ref"][:, 2],
        'vel_xr': log["vel_ref"][:, 0], 'vel_yr': log["vel_ref"][:, 1], 'vel_zr': log["vel_ref"][:, 2],
    })

if 'regressor' in controller.controller_flag:
    storage_dict.update({
        **{f'theta_{i}': log["theta_hat"][:, i] for i in range(log["theta_hat"].shape[1])}
    })

# 5. 创建并保存
df = pd.DataFrame(storage_dict)
df.to_csv(save_path, index=False)

print(f"Data saved to {save_path}")


####################################
# 保存为 .npz 文件
####################################
# save_path = sys.path[0] + '/data/RL_MRAC_flare.npz'

# 准备数据：将二维矩阵展平为一维数组
# kx_array = np.array([k.flatten() for k in log['kx']])          # (N, 18)
# kr_array = np.array([k.flatten() for k in log['kr']])          # (N, 9)
# theta_array = np.array([t.flatten() for t in log['theta']])    # (N, 18)
# thrust_array = np.array(log['thrust_pos'])                     # (N, 3)
# pos_err_array = np.array(log['pos_error'])                     # (N, 3)
# vel_err_array = np.array(log['vel_error'])                     # (N, 3)

# # 将数据保存为 .npz 文件
# np.savez(save_path,
#          kx=kx_array,
#          kr=kr_array,
#          theta=theta_array,
#          thrust=thrust_array,
#          pos_error=pos_err_array,
#          vel_error=vel_err_array)









