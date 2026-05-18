"""
组合损失 (Combined Criterion)
==============================
将三个损失统一管理:
  L_total = L_jepa + α(t) × L_topo + β × L_detect

其中 α(t) 是拓扑感知的自适应权重 (唯一入口, 不存在外部 scheduler):
  Phase 1 (warmup): 线性增长 init → base, 让模型先学基本对齐
  Phase 2 (adaptive): 根据 L_topo / L_jepa 的比值动态调节
    - 如果 L_topo 相对 L_jepa 过大 → 减小 α (防止拓扑主导)
    - 如果 L_topo 相对 L_jepa 过小 → 增大 α (强化拓扑约束)
    - 使用指数移动平均平滑比值, 避免震荡

这比纯线性 warmup 更符合 plan 中 "拓扑感知的自适应损失权重" 的设计:
  - plan line 92-95: "拓扑感知的自适应损失权重 ... 权重是自适应的"
  - 灵感来源: GradNorm (ICML 2018), 但更轻量 — 只用 loss ratio 不算梯度

数据契约:
  输入 model_output 来自 TopoJEPA.forward(), 包含:
    - "S_Y_hat": [B, D]
    - "S_Y": [B, D]
    - "topo_info": TopoInfo dict (TopologicalBranch 产出)
    - "feature_maps": List[Tensor]
    - "predictions": Optional[Dict] (DetectionHead 产出, Stage 2)

  输入 batch 来自 DataLoader (经 collate_fn, ultralytics 风格):
    - "img": [B, C, H, W]
    - "cls": [N_total, 1] (float tensor)
    - "bboxes": [N_total, 4] (cx, cy, w, h, 归一化)
    - "batch_idx": [N_total] (每个 obj 所属图片索引)
"""

import torch
import torch.nn as nn
from typing import Optional, Dict

from .jepa_loss import JEPALoss
from .topo_loss import TopologicalLoss
from .detection_loss import TopoDetectionLoss
from .seg_loss import SegmentationLoss


class TopoJEPACriterion(nn.Module):
    """
    TopoJEPA 统一损失函数

    L_total = L_jepa + α(t) × L_topo + β × L_task

    L_task 由 config 中 model.task_type 决定, 检测和分割互斥:
      - task_type="detection" (RDD): L_task = detect_weight × L_detect
      - task_type="segmentation" (DRIVE/Inria): L_task = seg_weight × L_seg

    互斥由 build_model / build_criterion 保证: 只构建其中一个 head/loss,
    另一个为 None。forward 中的运行时守卫是二重保险。

    训练阶段:
      Stage 1 (预训练): β = 0, 只有 L_jepa + α(t) × L_topo
      Stage 2 (微调):   L_jepa + α(t) × L_topo + L_task
    """

    def __init__(
        self,
        jepa_loss: JEPALoss,
        topo_loss: TopologicalLoss,
        detect_loss: Optional[TopoDetectionLoss] = None,
        seg_loss: Optional[SegmentationLoss] = None,
        # --- 拓扑权重自适应调度参数 ---
        topo_weight_init: float = 0.01,         # warmup 起点
        topo_weight_final: float = 1.0,         # warmup 终点 / 自适应基准值
        topo_warmup_steps: int = 1000,          # warmup 步数
        topo_adaptive: bool = True,             # 是否启用自适应调度 (否则退化为纯线性)
        topo_loss_ratio_target: float = 1.0,    # 目标 L_topo / L_jepa 比值
        topo_loss_ratio_ema: float = 0.99,      # 比值的 EMA 平滑系数
        topo_weight_min: float = 0.01,          # 自适应下限
        topo_weight_max: float = 5.0,           # 自适应上限
        topo_loss_max: float = 10.0,            # L_topo 值上限, 防止 PH 计算不稳定时梯度爆炸
        # --- 任务权重 ---
        jepa_weight: float = 1.0,
        detect_weight: float = 1.0,
        seg_weight: float = 1.0,
        # --- 实验模式 ---
        experiment_mode: str = "topojepa",     # topojepa | jepa_only | vicreg | barlow | toploss | topogcl
        training_stage: int = 1,
    ):
        """初始化组合损失"""
        super().__init__()
        self.jepa_loss = jepa_loss
        self.topo_loss = topo_loss
        self.detect_loss = detect_loss
        self.seg_loss = seg_loss

        # 拓扑权重调度参数
        self.topo_weight_init = topo_weight_init
        self.topo_weight_final = topo_weight_final
        self.topo_warmup_steps = topo_warmup_steps
        self.topo_adaptive = topo_adaptive
        self.topo_loss_ratio_target = topo_loss_ratio_target
        self.topo_loss_ratio_ema_coeff = topo_loss_ratio_ema
        self.topo_weight_min = topo_weight_min
        self.topo_weight_max = topo_weight_max
        self.topo_loss_max = topo_loss_max

        # 自适应状态 (non-parameter, 保存在 state_dict 中)
        self.register_buffer("_loss_ratio_ema", torch.tensor(topo_loss_ratio_target))
        self.register_buffer("_current_adaptive_weight", torch.tensor(topo_weight_final))

        # 任务权重 + 阶段
        self.jepa_weight = jepa_weight
        self.detect_weight = detect_weight
        self.seg_weight = seg_weight
        self.experiment_mode = experiment_mode
        self.training_stage = training_stage

    def get_topo_weight(
        self,
        step: int,
        loss_jepa: Optional[torch.Tensor] = None,
        loss_topo: Optional[torch.Tensor] = None,
    ) -> float:
        """
        拓扑权重 α(t) 的唯一计算入口 — 拓扑感知自适应调度

        两阶段策略:
          Phase 1 (step < warmup): 线性热身
            α = init + (final - init) * step / warmup
          Phase 2 (step >= warmup): 自适应调节 (若 topo_adaptive=True)
            1. 计算 loss ratio r = L_topo / L_jepa
            2. EMA 平滑: r_ema = ema_coeff * r_ema_prev + (1-ema_coeff) * r
            3. 目标: 让 r_ema ≈ target
               α = α_base * (target / r_ema)
               即 topo 过大 → α 缩小; topo 过小 → α 增大
            4. Clamp 到 [weight_min, weight_max]

        评估安全:
          当 self.training == False (即 model.eval() 后) 时,
          只读取已有状态, 不更新 _loss_ratio_ema / _current_adaptive_weight,
          避免 val 数据污染训练统计。

        Args:
            step: 当前训练步数
            loss_jepa: L_jepa (当前 step, detached), 自适应模式下需要
            loss_topo: L_topo (当前 step, detached, 未加权), 自适应模式下需要

        Returns:
            alpha: 当前拓扑权重
        """
        # Phase 1: 线性热身
        if step < self.topo_warmup_steps:
            ratio = step / max(self.topo_warmup_steps, 1)
            return self.topo_weight_init + (self.topo_weight_final - self.topo_weight_init) * ratio

        # Phase 2: 自适应 or 固定
        if not self.topo_adaptive or loss_jepa is None or loss_topo is None:
            return self.topo_weight_final

        # 计算 loss ratio (detach, 避免影响计算图)
        l_jepa_val = loss_jepa.detach().item()
        l_topo_val = loss_topo.detach().item()

        if l_jepa_val < 1e-8:
            return self._current_adaptive_weight.item()

        current_ratio = l_topo_val / l_jepa_val

        # 评估模式: 只读, 不写回状态
        if not self.training:
            ema_val = max(self._loss_ratio_ema.item(), 1e-8)
            adaptive_weight = self.topo_weight_final * (self.topo_loss_ratio_target / ema_val)
            return max(self.topo_weight_min, min(self.topo_weight_max, adaptive_weight))

        # 训练模式: 更新 EMA 状态
        self._loss_ratio_ema.mul_(self.topo_loss_ratio_ema_coeff).add_(
            (1 - self.topo_loss_ratio_ema_coeff) * current_ratio)

        # 自适应调节: α = α_base × (target / r_ema)
        ema_val = max(self._loss_ratio_ema.item(), 1e-8)
        adaptive_weight = self.topo_weight_final * (self.topo_loss_ratio_target / ema_val)

        # Clamp
        adaptive_weight = max(self.topo_weight_min, min(self.topo_weight_max, adaptive_weight))

        self._current_adaptive_weight.fill_(adaptive_weight)
        return adaptive_weight

    def forward(
        self,
        model_output: Dict,
        batch: Optional[Dict] = None,
        step: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """
        计算组合损失

        Returns:
            dict with:
              - "loss": 总损失 (标量, 用于 backward)
              - "loss_jepa": JEPA 损失
              - "loss_topo": 拓扑损失 (未加权, 供监控)
              - "loss_detect": 检测损失 (Stage 2)
              - "topo_weight": α(t) (当前自适应权重)
              - "loss_ratio_ema": EMA 平滑后的 L_topo/L_jepa 比值 (监控)
              - 以及各子损失的详细分项
        """
        result = {}

        # --- L_jepa / experiment modes ---
        # 标准 JEPA: S_Y_hat 和 S_Y 都是视觉嵌入, 不再需要 target_texts 做语义分组
        jepa_out = self.jepa_loss(
            model_output["S_Y_hat"],
            model_output["S_Y"],
        )
        result["loss_jepa"] = jepa_out["loss"]
        result.update({f"jepa/{k}": v for k, v in jepa_out.items() if k != "loss"})

        # baseline experiments
        if self.experiment_mode == "vicreg":
            result["loss_jepa"] = result["loss_jepa"] + self.jepa_loss.reg_coeff * self.jepa_loss.variance_regularization(model_output["S_Y_hat"])
        elif self.experiment_mode == "barlow":
            z1 = torch.nn.functional.normalize(model_output["S_Y_hat"].float(), dim=-1)
            z2 = torch.nn.functional.normalize(model_output["S_Y"].float().detach(), dim=-1)
            c = (z1.T @ z2) / z1.shape[0]
            on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
            off_diag = (c - torch.diag(torch.diagonal(c))).pow(2).sum()
            result["loss_jepa"] = on_diag + 0.005 * off_diag
        elif self.experiment_mode == "topogcl":
            z = torch.nn.functional.normalize(model_output["S_Y_hat"].float(), dim=-1)
            sim = z @ z.T / 0.2
            labels = torch.arange(z.shape[0], device=z.device)
            result["loss_jepa"] = torch.nn.functional.cross_entropy(sim, labels)
        elif self.experiment_mode == "toploss":
            pass
        elif self.experiment_mode == "jepa_only":
            pass

        # --- L_topo (消费 topo_info, 不重复计算 PH) ---
        raw_topo = torch.tensor(0.0, device=result["loss_jepa"].device)
        if self.experiment_mode not in ("jepa_only", "barlow", "vicreg"):
            topo_out = self.topo_loss(model_output["topo_info"])
            raw_topo = topo_out["loss"]
            result.update({f"topo/{k}": v for k, v in topo_out.items() if k != "loss"})
        else:
            topo_out = {"loss": raw_topo}
        result["loss_topo"] = torch.clamp(raw_topo, max=self.topo_loss_max)

        if self.experiment_mode == "topojepa" and self.training_stage in (1, 2, 12):
            result["topo/fidelity"] = topo_out.get("loss_fidelity", torch.tensor(0.0, device=raw_topo.device))
            result["topo/preserve"] = topo_out.get("loss_preserve", torch.tensor(0.0, device=raw_topo.device))
            result["topo/collapse"] = topo_out.get("loss_collapse", torch.tensor(0.0, device=raw_topo.device))

        # --- 拓扑感知自适应权重 ---
        alpha = 0.0 if self.experiment_mode in ("jepa_only", "barlow", "vicreg") else self.get_topo_weight(
            step=step,
            loss_jepa=result["loss_jepa"],
            loss_topo=result["loss_topo"],
        )
        result["topo_weight"] = alpha
        result["loss_ratio_ema"] = self._loss_ratio_ema.item()

        # --- L_detect (Stage 2 / 12, RDD 检测) ---
        loss_detect = torch.tensor(0.0, device=result["loss_jepa"].device)
        if self.training_stage in (2, 12) and self.detect_loss is not None:
            predictions = model_output.get("predictions")
            if predictions is not None and batch is not None:
                detect_out = self.detect_loss(predictions, batch)
                loss_detect = detect_out["loss"]
                result.update({f"detect/{k}": v for k, v in detect_out.items() if k != "loss"})
        result["loss_detect"] = loss_detect

        # --- L_seg (Stage 2 / 12, DRIVE/Inria 分割) ---
        loss_seg = torch.tensor(0.0, device=result["loss_jepa"].device)
        if self.training_stage in (2, 12) and self.seg_loss is not None:
            seg_predictions = model_output.get("seg_predictions")
            if seg_predictions is not None and batch is not None and batch.get("masks") is not None:
                seg_out = self.seg_loss(seg_predictions, batch)
                loss_seg = seg_out["loss"]
                result.update({f"seg/{k}": v for k, v in seg_out.items() if k != "loss"})
        result["loss_seg"] = loss_seg

        # --- Total ---
        result["loss"] = (
            self.jepa_weight * result["loss_jepa"]
            + alpha * result["loss_topo"]
            + self.detect_weight * result["loss_detect"]
            + self.seg_weight * result["loss_seg"]
        )
        return result

    def set_stage(self, stage: int) -> None:
        """
        切换训练阶段
          stage=1: 预训练 (无检测损失)
          stage=2: 微调 (全损失)
        """
        self.training_stage = stage
