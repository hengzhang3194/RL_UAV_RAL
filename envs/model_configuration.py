import numpy as np
from dataclasses import dataclass

@dataclass
class DroneConfig:
    """
    平级配置类：整合所有物理参数与计算逻辑。
    可以直接通过 self.mass = config.mass 这种方式在 Env 中解构。
    """
    # --- 基础输入参数 ---
    model_name: str
    mass: float
    inertial: np.ndarray
    hovering_throttle: float
    duration: float = 30.0
    position_frequency: float = 50.0
    attitude_frequency: float = 200.0
    g: float = 9.81

    # 仅保留你给出的计算逻辑
    def __post_init__(self):
        # 计算控制采样间隔
        self.dt = 1.0 / self.attitude_frequency
        self.dt_pos = 1.0 / self.position_frequency
        # 计算位置环与姿态环的频率倍数
        self.pos_att_power = round(self.attitude_frequency / self.position_frequency)
        self.POTT = self.hovering_throttle / (self.mass * self.g)

    # --- 静态方法：快速获取不同机型的实例 ---
    @staticmethod
    def get_model(model_name='P600', duration=30.0):
        if model_name == 'P600':
            mass = 3.3
            inertial = np.diag([0.05487, 0.05487, 0.1027])
            hovering_throttle = 0.5
        elif model_name == 'M0':
            mass = 1.32
            inertial = np.diag([0.003686, 0.003686, 0.006824])
            hovering_throttle = 0.4
        
        # 修复：将提取的参数填入 DroneConfig 实例化
        return DroneConfig(
            model_name=model_name,
            mass=mass,
            inertial=inertial,
            hovering_throttle=hovering_throttle,
            duration=duration
        )

# 验证运行
if __name__ == "__main__":
    config = DroneConfig.get_model('P600')
    print(f"成功加载: {config.model_name}")
    print(f"采样间隔 dt: {config.dt}")
    # 此时 config.POTT 不存在，符合你最后给出的要求