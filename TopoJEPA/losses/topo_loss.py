"""
拓扑损失 (Topological Loss)
============================
本文最核心的贡献: 可微的嵌入空间拓扑约束

计算流程:
  TopologicalBranch (models/topo_branch.py) 持有 DifferentiablePH, 计算 diagrams
  → 本模块消费 diagrams → 输出损失
  本模块 **不** 从 raw embedding 重复计算 PH
  本模块 **不** 持有 DifferentiablePH (它在 topo_branch.py 中)

数据契约:
  - PersistenceDiagram = Tuple[Tensor births [K], Tensor deaths [K]]
  - 输入: TopoInfo dict, 来自 TopologicalBranch.forward()
  - 输出: 损失 dict

Sliced Wasserstein 距离参考:
  Carrière, Cuturi, Oudot (2017) "Sliced Wasserstein Kernel for Persistence Diagrams"
  - 将 PD 视为 2D 平面 (birth, death) 上的点集
  - 每个点的对角投影 (birth+death)/2 是其"消亡副本"
  - 用随机 1D 方向投影, 排序后求 L^p 距离
  - 复杂度 O(N log N) per slice, 共 num_slices 个方向

性能优化:
  - 使用 Sliced Wasserstein 距离近似 O(N log N)
"""

import math
import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Dict

# 与 models/topo_branch.py 共享类型定义
PersistenceDiagram = Tuple[torch.Tensor, torch.Tensor]  # (births [K], deaths [K])

_EPS = 1e-7


def _pd_to_points(diagram: PersistenceDiagram) -> torch.Tensor:
    """
    将 PersistenceDiagram 转为 2D 点集 [K, 2] (birth, death)
    过滤掉 persistence ≤ 0 的点 (placeholder / degenerate)
    """
    births, deaths = diagram
    pers = deaths - births
    # 保留 persistence > eps 的点
    mask = pers > _EPS
    if mask.sum() == 0:
        # 返回空 diagram (对角线上一点, 不贡献距离)
        return torch.zeros(1, 2, device=births.device)
    return torch.stack([births[mask], deaths[mask]], dim=-1)  # [K', 2]


def _augment_with_diagonal(pts: torch.Tensor, other_size: int) -> torch.Tensor:
    """
    标准做法: 对方有 M 个点, 我有 N 个点
    需要给我方添加 M 个对角投影 (来自对方的点的对角投影)
    但这里我们用更简洁的做法: 双方都补对角投影到相同大小
    """
    # 对角投影: (birth, death) → ((birth+death)/2, (birth+death)/2)
    diag = pts.sum(dim=-1, keepdim=True) / 2.0  # [N, 1]
    diag_pts = diag.expand(-1, 2)               # [N, 2]
    return diag_pts


class TopologicalLoss(nn.Module):
    """
    拓扑损失: 消费 TopologicalBranch 的输出, 计算损失

    三个损失分项 (可配置启用哪些):
      1. L_fidelity: 预测嵌入和目标嵌入的持久图应一致
         W_p(pred_diagrams, target_diagrams) → min
      2. L_preserve: 嵌入应保持特征图的拓扑结构
         Σ_k |β_k(feature) - β_k(embedding)| → min
      3. L_collapse: 嵌入空间不应拓扑坍塌
         -TotalPersistence(pred_diagrams) → min (最大化总持久性)

    输入: TopoInfo dict (来自 TopologicalBranch.forward())
    输出: 损失 dict
    """

    def __init__(
        self,
        loss_components: Optional[List[str]] = None,  # ["fidelity", "preserve", "collapse"]
        wasserstein_order: int = 2,
        use_sliced_wasserstein: bool = True,
        num_slices: int = 50,
        fidelity_weight: float = 1.0,
        preserve_weight: float = 0.5,
        collapse_weight: float = 0.5,
        persistence_threshold: float = 0.01,           # 须与 TopologicalBranch 一致
    ):
        """初始化拓扑损失"""
        super().__init__()
        self.loss_components = loss_components or ["fidelity", "preserve", "collapse"]
        self.wasserstein_order = wasserstein_order
        self.use_sliced_wasserstein = use_sliced_wasserstein
        self.num_slices = num_slices
        self.fidelity_weight = fidelity_weight
        self.preserve_weight = preserve_weight
        self.collapse_weight = collapse_weight
        self.persistence_threshold = persistence_threshold

    # ============ Wasserstein 距离 ============

    def wasserstein_distance(
        self,
        diagram_a: PersistenceDiagram,
        diagram_b: PersistenceDiagram,
    ) -> torch.Tensor:
        """
        计算两个持久图之间的 Wasserstein 距离
        根据 use_sliced_wasserstein 选择精确或近似算法

        Returns:
            distance: 标量 tensor
        """
        if self.use_sliced_wasserstein:
            return self.sliced_wasserstein_distance(diagram_a, diagram_b)
        return self.exact_wasserstein_distance(diagram_a, diagram_b)

    def exact_wasserstein_distance(
        self,
        diagram_a: PersistenceDiagram,
        diagram_b: PersistenceDiagram,
    ) -> torch.Tensor:
        """
        精确 p-Wasserstein 距离 (适用于小规模 PD)

        算法:
          1. 将 PD 转为 2D 点集, 双方互补对角投影 (同 SWD)
          2. 用 cost matrix + 贪心/排序匹配 (近似)
             对于精确解需要匈牙利算法, 但 torch 无内置实现
             这里用 persistence 排序匹配: 按 persistence 降序排, 一一配对
             这在多数实际 PD 上是最优或接近最优的

        Returns:
            distance: 标量 tensor (可微)
        """
        pts_a = _pd_to_points(diagram_a)  # [Na, 2]
        pts_b = _pd_to_points(diagram_b)  # [Nb, 2]
        device = pts_a.device

        # 互补对角投影, 使两方大小相同 = Na + Nb
        diag_a = _augment_with_diagonal(pts_a, pts_b.shape[0])
        diag_b = _augment_with_diagonal(pts_b, pts_a.shape[0])
        pts_a_aug = torch.cat([pts_a, diag_b], dim=0)  # [Na+Nb, 2]
        pts_b_aug = torch.cat([pts_b, diag_a], dim=0)  # [Na+Nb, 2]

        # 按 persistence 降序排序后匹配
        pers_a = (pts_a_aug[:, 1] - pts_a_aug[:, 0]).abs()
        pers_b = (pts_b_aug[:, 1] - pts_b_aug[:, 0]).abs()
        order_a = pers_a.argsort(descending=True)
        order_b = pers_b.argsort(descending=True)

        pts_a_sorted = pts_a_aug[order_a]
        pts_b_sorted = pts_b_aug[order_b]

        # L^p 距离
        p = self.wasserstein_order
        diff = (pts_a_sorted - pts_b_sorted).abs()  # [N, 2]
        if p == 1:
            cost = diff.sum(dim=-1)  # L1 on R^2
        elif p == 2:
            cost = (diff ** 2).sum(dim=-1).sqrt()  # L2 on R^2
        else:
            cost = (diff ** p).sum(dim=-1) ** (1.0 / p)

        return cost.mean()

    def sliced_wasserstein_distance(
        self,
        diagram_a: PersistenceDiagram,
        diagram_b: PersistenceDiagram,
    ) -> torch.Tensor:
        """
        Sliced Wasserstein 距离: O(N log N) 近似

        算法:
          1. 将 PD 转为 2D 点集 (birth, death)
          2. 双方互补对角投影 (标准 Wasserstein on PD 做法)
          3. 生成 num_slices 个随机方向 θ ∈ [0, π)
          4. 投影到每个方向, 排序, 求 L^p 距离
          5. 取平均

        Returns:
            distance: 标量 tensor (可微)
        """
        pts_a = _pd_to_points(diagram_a)  # [Na, 2]
        pts_b = _pd_to_points(diagram_b)  # [Nb, 2]
        device = pts_a.device

        # 标准做法: 每方把对方的点的对角投影加入自己
        # A' = A ∪ diag(B),  B' = B ∪ diag(A)
        diag_a = _augment_with_diagonal(pts_a, pts_b.shape[0])  # [Na, 2]
        diag_b = _augment_with_diagonal(pts_b, pts_a.shape[0])  # [Nb, 2]

        pts_a_aug = torch.cat([pts_a, diag_b], dim=0)  # [Na+Nb, 2]
        pts_b_aug = torch.cat([pts_b, diag_a], dim=0)  # [Nb+Na, 2]

        # 生成随机方向 (均匀分布在半圆 [0, π))
        angles = torch.linspace(0, math.pi, self.num_slices + 1, device=device)[:-1]
        # 方向向量 [num_slices, 2]
        directions = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)

        # 投影: [num_slices, Na+Nb]
        proj_a = pts_a_aug @ directions.T  # [Na+Nb, num_slices] -> 转置
        proj_b = pts_b_aug @ directions.T  # [Na+Nb, num_slices]

        # 排序
        proj_a_sorted, _ = proj_a.sort(dim=0)
        proj_b_sorted, _ = proj_b.sort(dim=0)

        # L^p 距离
        p = self.wasserstein_order
        if p == 1:
            dist = (proj_a_sorted - proj_b_sorted).abs().mean()
        elif p == 2:
            dist = ((proj_a_sorted - proj_b_sorted) ** 2).mean().sqrt()
        else:
            dist = ((proj_a_sorted - proj_b_sorted).abs() ** p).mean() ** (1.0 / p)

        return dist

    # ============ 三个损失分项 ============

    def _betti_curve_distance(
        self,
        diagram_a: PersistenceDiagram,
        diagram_b: PersistenceDiagram,
        grid: torch.Tensor,
    ) -> torch.Tensor:
        """
        归一化 Betti 曲线的 L1 距离近似。

        β(λ) = #{(b, d) : b <= λ < d}
        这里用 soft counting 近似阶跃，便于反传。
        """
        births_a, deaths_a = diagram_a
        births_b, deaths_b = diagram_b

        pers_a = (deaths_a - births_a).clamp(min=0)
        pers_b = (deaths_b - births_b).clamp(min=0)
        if pers_a.numel() == 0 or pers_b.numel() == 0:
            return torch.tensor(0.0, device=grid.device)

        tau = max(self.persistence_threshold, 1e-6)
        temp = max(tau * 0.25, 1e-3)

        def _soft_betti(births: torch.Tensor, deaths: torch.Tensor) -> torch.Tensor:
            g = grid.view(1, -1)
            b = births.view(-1, 1)
            d = deaths.view(-1, 1)
            alive = torch.sigmoid((g - b) / temp) * torch.sigmoid((d - g) / temp)
            return alive.sum(dim=0)

        beta_a = _soft_betti(births_a, deaths_a)
        beta_b = _soft_betti(births_b, deaths_b)
        denom = (beta_a.abs().mean() + beta_b.abs().mean()).clamp(min=_EPS)
        return torch.mean(torch.abs(beta_a - beta_b)) / denom

    def topological_fidelity_loss(
        self,
        pred_diagrams: List[PersistenceDiagram],
        target_diagrams: List[PersistenceDiagram],
    ) -> torch.Tensor:
        """
        拓扑保真损失: 预测和目标的持久图应一致

        L_fidelity = Σ_dim W_p(pred_diagrams[dim], target_diagrams[dim])
        """
        device = pred_diagrams[0][0].device
        total = torch.tensor(0.0, device=device)
        n_dims = min(len(pred_diagrams), len(target_diagrams))
        for dim in range(n_dims):
            total = total + self.wasserstein_distance(
                pred_diagrams[dim], target_diagrams[dim])
        return total

    def topological_preservation_loss(
        self,
        pred_diagrams: List[PersistenceDiagram],
        feature_diagrams: List[PersistenceDiagram],
    ) -> torch.Tensor:
        """
        拓扑保持损失: 对齐嵌入空间与特征空间的归一化 Betti 曲线。

        这更贴近论文中的 L_preserve = Σ_k mean_λ |β̄_feat - β̄_emb|。
        这里保留一个轻量 PD 距离作为辅助项，但主项是 Betti 曲线对齐。
        """
        device = pred_diagrams[0][0].device
        total = torch.tensor(0.0, device=device)
        n_dims = min(len(pred_diagrams), len(feature_diagrams))
        grid = torch.linspace(0.0, 1.0, 32, device=device)
        for dim in range(n_dims):
            betti_gap = self._betti_curve_distance(pred_diagrams[dim], feature_diagrams[dim], grid)
            pd_gap = self.wasserstein_distance(pred_diagrams[dim], feature_diagrams[dim])
            total = total + betti_gap + 0.25 * pd_gap
        return total

    def topological_anti_collapse_loss(
        self,
        pred_diagrams: List[PersistenceDiagram],
        target_diagrams: Optional[List[PersistenceDiagram]] = None,
    ) -> torch.Tensor:
        """
        拓扑反坍塌损失: 使用论文中的 persistence floor τ_k 近似实现。

        论文定义: L_collapse = Σ_k max(0, τ_k - TP_k(PD_k(pred)))
        这里用 target diagrams 的均值总持久性作为 batch-wise τ_k 估计。
        """
        device = pred_diagrams[0][0].device
        total = torch.tensor(0.0, device=device)
        for idx, (births, deaths) in enumerate(pred_diagrams):
            pred_tp = torch.clamp(deaths - births, min=0).sum()
            tau_k = pred_tp.detach() * 0.0
            if target_diagrams is not None and idx < len(target_diagrams):
                tb, td = target_diagrams[idx]
                tau_k = torch.clamp(td - tb, min=0).sum().detach()
            total = total + torch.relu(tau_k - pred_tp)
        return total

    # ============ 主前向 ============

    def forward(
        self,
        topo_info: Dict,
    ) -> Dict[str, torch.Tensor]:
        """
        计算总拓扑损失

        Args:
            topo_info: TopoInfo dict from TopologicalBranch.forward()

        Returns:
            dict with: loss, loss_fidelity, loss_preserve, loss_collapse,
                        betti_pred, betti_target
        """
        losses = {}
        total = torch.tensor(0.0, device=topo_info["pred_betti"].device)

        if "fidelity" in self.loss_components:
            l_fid = self.topological_fidelity_loss(
                topo_info["pred_diagrams"], topo_info["target_diagrams"])
            losses["loss_fidelity"] = l_fid
            total = total + self.fidelity_weight * l_fid

        if "preserve" in self.loss_components and topo_info.get("feature_diagrams") is not None:
            l_pres = self.topological_preservation_loss(
                topo_info["pred_diagrams"], topo_info["feature_diagrams"])
            losses["loss_preserve"] = l_pres
            total = total + self.preserve_weight * l_pres

        if "collapse" in self.loss_components:
            l_col = self.topological_anti_collapse_loss(
                topo_info["pred_diagrams"], topo_info.get("target_diagrams")
            )
            losses["loss_collapse"] = l_col
            total = total + self.collapse_weight * l_col

        losses["loss"] = total
        losses["betti_pred"] = topo_info["pred_betti"]
        losses["betti_target"] = topo_info["target_betti"]
        return losses
