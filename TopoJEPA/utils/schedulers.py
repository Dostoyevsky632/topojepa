"""
调度器
======
学习率和权重衰减的调度

注意: 拓扑权重 α(t) 的调度已统一封装在 TopoJEPACriterion.get_topo_weight() 中
      本文件不提供拓扑权重调度器, 避免双入口冲突
"""

import math


class CosineScheduler:
    """
    余弦调度器 (通用: 学习率 / 权重衰减 等)

    参考 jepa/src/utils/schedulers.py
    支持线性热身 + 余弦衰减:
      t < warmup: value = warmup_value + (init - warmup_value) * t / warmup
      t >= warmup: value = final + 0.5 * (init - final) * (1 + cos(pi * (t-warmup) / (T-warmup)))
    """

    def __init__(
        self,
        init_value: float,
        final_value: float,
        total_steps: int,
        warmup_steps: int = 0,
        warmup_value: float = 0.0,
    ):
        """初始化调度器"""
        self.init_value = init_value
        self.final_value = final_value
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.warmup_value = warmup_value

    def step(self, current_step: int) -> float:
        """返回当前步数的值"""
        if current_step < self.warmup_steps:
            # 线性热身
            ratio = current_step / max(self.warmup_steps, 1)
            return self.warmup_value + (self.init_value - self.warmup_value) * ratio
        else:
            # 余弦衰减
            progress = (current_step - self.warmup_steps) / max(
                self.total_steps - self.warmup_steps, 1)
            progress = min(progress, 1.0)
            return self.final_value + 0.5 * (self.init_value - self.final_value) * (
                1.0 + math.cos(math.pi * progress))
