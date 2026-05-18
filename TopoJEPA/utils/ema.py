"""
EMA (Exponential Moving Average)
================================
参考 jepa/app/vjepa/train.py 的动量更新机制

JEPA 范式中 target encoder 的更新方式:
  target_param = m * target_param + (1 - m) * source_param
  动量 m 从 ema_start 线性增长到 ema_end (趋向 1.0)
"""

import copy
import torch
import torch.nn as nn
from typing import Tuple


class ExponentialMovingAverage:
    """
    Target encoder 的 EMA 更新器

    动量调度: m 从 ema_start 线性增长到 ema_end
    参考 jepa/app/vjepa/train.py line 302-303:
      momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(ipe*num_epochs))
    """

    def __init__(
        self,
        source_model: nn.Module,              # 源模型 (encoder/predictor)
        target_model: nn.Module,              # 目标模型 (target_encoder)
        ema_range: Tuple[float, float] = (0.996, 1.0),
        total_steps: int = 100000,
    ):
        """初始化 EMA"""
        self.source_model = source_model
        self.target_model = target_model
        self.ema_start, self.ema_end = ema_range
        self.total_steps = max(total_steps, 1)

    def get_momentum(self, step: int) -> float:
        """
        获取当前步数的动量值

        线性插值: m(step) = ema_start + (ema_end - ema_start) * step / total_steps
        Clamp 到 [ema_start, ema_end]
        """
        ratio = min(step / self.total_steps, 1.0)
        return self.ema_start + (self.ema_end - self.ema_start) * ratio

    @torch.no_grad()
    def update(self, step: int) -> float:
        """
        执行一步 EMA 更新
        target = m * target + (1-m) * source

        target_model 始终保持 eval 模式 (防御性)

        Returns:
            momentum: 使用的动量值
        """
        m = self.get_momentum(step)
        for p_src, p_tgt in zip(self.source_model.parameters(),
                                self.target_model.parameters()):
            p_tgt.data.mul_(m).add_(p_src.data, alpha=1.0 - m)
        # 防御性: 确保 target 始终 eval
        if self.target_model.training:
            self.target_model.eval()
        return m

    @staticmethod
    def build_target(source_model: nn.Module) -> nn.Module:
        """
        从源模型构建 target 模型 (深拷贝 + 冻结梯度 + eval 模式)

        JEPA 的 target encoder 必须始终处于 eval 模式:
          - 关闭 dropout / batch norm 训练态
          - 确保目标嵌入稳定 (稳定 teacher)
        """
        target = copy.deepcopy(source_model)
        for p in target.parameters():
            p.requires_grad_(False)
        target.eval()
        return target
