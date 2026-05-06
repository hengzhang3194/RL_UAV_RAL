import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import os

# --- 全局字体与大小设置 (学术论文出版标准) ---
FONT_SIZE_MAIN = 28
FONT_SIZE_TITLE = 24
FONT_SIZE_TICK = 28
FONT_SIZE_LEGEND = 28

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.labelsize'] = FONT_SIZE_MAIN    
plt.rcParams['axes.titlesize'] = FONT_SIZE_TITLE    
plt.rcParams['xtick.labelsize'] = FONT_SIZE_TICK   
plt.rcParams['ytick.labelsize'] = FONT_SIZE_TICK   
plt.rcParams['legend.fontsize'] = FONT_SIZE_LEGEND   

def plot_z_with_inset_zoom(base_dir, file_names, data_range=(201, 600)):
    """
    绘制 Z 轴均值方差图，包含局部放大子图、地面标注和地面效应区
    """
    all_z_actual = []
    all_times = []
    min_len = float('inf')

    # 1. 加载数据
    for name in file_names:
        file_path = os.path.join(base_dir, name)
        if not os.path.exists(file_path):
            print(f"跳过：找不到文件 {file_path}")
            continue
        try:
            df = pd.read_csv(file_path)
            z_act = df["pos_z"].iloc[data_range[0]:data_range[1]].to_numpy()
            t_raw = df["time"].iloc[data_range[0]:data_range[1]].to_numpy()
            
            all_z_actual.append(z_act)
            all_times.append(t_raw - t_raw[0])
            min_len = min(min_len, len(z_act))
        except Exception as e:
            print(f"读取 {name} 时出错: {e}")

    if not all_z_actual:
        print("错误：未加载到任何有效数据。")
        return

    z_matrix = np.array([z[:min_len] for z in all_z_actual])
    t_matrix = np.array([t[:min_len] for t in all_times])
    z_mean = np.mean(z_matrix, axis=0)
    z_std = np.std(z_matrix, axis=0)
    time_axis = np.mean(t_matrix, axis=0)

    # 3. 创建 16:9 主图画布
    fig, ax = plt.subplots(figsize=(16, 9), dpi=100)
    
    ground_z = 0.178     
    ge_threshold = 0.24  
    des_z = 0.2          

    def draw_content(target_ax, is_inset=False):
        """主图与子图共用的绘图逻辑"""
        target_ax.axhspan(ground_z, ge_threshold, facecolor='tab:green', alpha=0.15)
        target_ax.axhline(y=des_z, color='green', linestyle='--', linewidth=3)
        target_ax.axhline(y=ground_z, color='red', linestyle='-', linewidth=4)
        target_ax.fill_between(time_axis, z_mean - z_std, z_mean + z_std, color='tab:blue', alpha=0.2)
        target_ax.plot(time_axis, z_mean, color='tab:blue', linewidth=3.5)
        
        if not is_inset:
            target_ax.axhspan(-1, ground_z, facecolor='dimgray', alpha=0.7, hatch='\\\\\\')
            target_ax.text(time_axis[0] + 0.05, ground_z - 0.005, '', color='white', 
                           fontweight='bold', fontsize=18, va='top', bbox=dict(facecolor='red', alpha=0.8, edgecolor='none'))

    draw_content(ax)

    # 4. 创建局部放大子图
    ax_ins = inset_axes(ax, width="45%", height="40%", loc='lower left', 
                        bbox_to_anchor=(0.45, 0.26, 1, 1), bbox_transform=ax.transAxes)
    
    draw_content(ax_ins, is_inset=True)

    zoom_start_idx = int(len(time_axis) * 0.4) 
    zoom_end_idx = int(len(time_axis) * 0.95) 
    ax_ins.set_xlim(time_axis[zoom_start_idx], time_axis[zoom_end_idx])
    ax_ins.set_ylim(ground_z - 0.01, 0.3) 
    
    ax_ins.tick_params(axis='both', which='major', labelsize=FONT_SIZE_TICK, direction='in', length=6)
    for spine in ax_ins.spines.values():
        spine.set_linewidth(2) 
    ax_ins.grid(True, linestyle=':', alpha=0.6)
    
    # --- 修正后的 mark_inset：针对 TypeError 的鲁棒处理 ---
    # 强制不使用内置样式，手动获取返回对象
    result = mark_inset(ax, ax_ins, loc1=1, loc2=1, fc="none", ec="black", lw=2.5)
    
    # 获取连接线对象 (通常在返回值的第 2 个元素)
    # 处理不同版本返回 2 个或 3 个值的情况
    lines = result[1] 
    
    # 判断 lines 是否为可迭代对象 (解决 TypeError)
    if not hasattr(lines, '__iter__'):
        lines = [lines] # 如果是单个对象，包装成列表
    
    for line in lines:
        line.set_linestyle('--') 
        line.set_linewidth(2.5)  

    # 5. 主图完善修饰
    ax.set_title("UAV Landing Performance - Ground Effect Analysis", pad=25, fontweight='bold')
    ax.set_xlabel("Time [s]", labelpad=12)
    ax.set_ylabel("Altitude Z [m]", labelpad=12)
    ax.grid(True, linestyle=':', alpha=0.7)
    
    ax.set_ylim(ground_z - 0.05, np.max(z_mean) + 0.1)
    ax.set_xlim(time_axis[0], time_axis[-1])
    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True, length=8)

    # 自定义图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='tab:blue', lw=3.5, label='Actual Height (Mean)'),
        Line2D([0], [0], color='green', lw=3, ls='--', label=f'Desired ({des_z}m)'),

    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=True, shadow=True, borderpad=1)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    target_dir = r"E:\Zhangheng\RL_MRAC_RAL\Data"
    target_files = ["test0.csv", "test1.csv", "test2.csv", "test3.csv", "test4.csv"]
    
    plot_z_with_inset_zoom(target_dir, target_files)