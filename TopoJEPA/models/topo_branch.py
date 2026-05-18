"""
拓扑分支 (Topological Branch)
=============================
本文的核心创新之一: 在嵌入空间中计算拓扑特征, 用于正则化

设计理念:
  传统 TopoLoss 在像素空间用 Cubical Complex 计算持久同调
  我们在嵌入空间用 Rips Complex 计算持久同调
  因为嵌入是高维点云, 不是 2D 网格

两层拓扑约束:
  1. 嵌入空间拓扑: batch 内的 B 个 D 维嵌入向量视为点云
     → 用 DifferentiablePH (Rips filtration) 计算持久同调
  2. 特征空间拓扑: 从 encoder 的 2D 特征图提取拓扑签名
     → 用 DifferentiablePH (Cubical filtration) 计算

组件关系:
  DifferentiablePH — 底层可微 PH 引擎 (定义在本文件中)
       ↑ 被持有和调用
  TopologicalBranch — 产出 TopoInfo dict
       ↓ 被消费
  TopologicalLoss  — 消费 diagrams, 计算损失 (定义在 losses/topo_loss.py)

维度约定 (与 topojepa.py 的数据契约一致):
  - 嵌入输入: [B, D] (池化后的嵌入向量, 不是 [B, N, D])
  - 特征图输入: [B, C, H, W]
  - diagram 输出: PersistenceDiagram = Tuple[Tensor births [K], Tensor deaths [K]]
  - TopoInfo 输出: 见 topojepa.py 顶部契约

职责边界:
  - TopologicalBranch 负责: 持有 DifferentiablePH, 计算 PH, 产出 diagrams + Betti 数
  - TopologicalLoss 负责: 消费 diagrams, 计算 Wasserstein 距离, 输出损失
  - 不重复计算: loss 侧不再从 raw embedding 重新算 PH

可微 PH 实现策略:
  1. 距离矩阵: 可微欧氏距离, 零点加 eps 避免 NaN 梯度
  2. Rips H0: Kruskal MST — Union-Find 决定结构, 边权保持可微
  3. Rips H1: 非 MST 边的 birth = 边权; death = 该边两端点在 MST 上的
     最长路径边权 (bottleneck path), 纯 torch 可微
  4. Cubical H0: sublevel set Union-Find (elder rule)
  5. Cubical H1: 非树边在 4-邻域生成树上形成环路 (简版)
  6. 后备: 若 giotto-tda 可用则用精确 Rips + 梯度穿透
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Dict

# 标准化类型别名
PersistenceDiagram = Tuple[torch.Tensor, torch.Tensor]  # (births [K], deaths [K])

# 数值安全常量
_EPS = 1e-7


# ============================================================
# 底层可微 PH 引擎
# ============================================================

class DifferentiablePH(nn.Module):
    """
    可微持久同调计算模块

    将不可微的 PH 计算转化为可参与梯度传播的操作:
      1. 从嵌入计算距离矩阵 (可微)
      2. 用 Rips/Cubical filtration
      3. 输出 PersistenceDiagram (births, deaths tensors)

    支持两种 filtration:
      - "rips": 用于高维嵌入点云 [N, D]
      - "cubical": 用于 2D 特征图 [H, W]

    后备策略:
      - 尝试 import giotto-tda, 委托精确 Rips 计算 + 梯度穿透
      - 失败则用纯 PyTorch 实现 (保证可运行, 梯度可回传)
    """

    def __init__(
        self,
        max_homology_dim: int = 1,
        max_points: int = 128,
        filtration_type: str = "rips",
    ):
        super().__init__()
        self.max_homology_dim = max_homology_dim
        self.max_points = max_points
        self.filtration_type = filtration_type

        # 探测可用后端
        self._has_giotto = self._check_import("gtda")
        self._has_gudhi = self._check_import("gudhi")

    @staticmethod
    def _check_import(module_name: str) -> bool:
        try:
            __import__(module_name)
            return True
        except ImportError:
            return False

    # ==================== 距离矩阵 ====================

    def compute_distance_matrix(
        self,
        points: torch.Tensor,                 # [N, D]
    ) -> torch.Tensor:
        """
        可微欧氏距离矩阵, 零点安全

        使用 torch.cdist (内部优化, 避免 [N,N,D] 中间张量):
          - 比手动广播 unsqueeze(0)-unsqueeze(1) 快 2-3x, 显存减半
          - 对角线 clamp 到 0, 重合点 clamp 到 eps

        Returns:
            dist_matrix: [N, N]
        """
        N = points.shape[0]
        # torch.cdist: 高效成对距离, 内部用 BLAS 优化
        # 输入需要 3D: [1, N, D] → 输出 [1, N, N] → squeeze
        dist = torch.cdist(points.unsqueeze(0), points.unsqueeze(0)).squeeze(0)  # [N, N]
        # 零点安全: 重合点距离 clamp 到 eps (避免 backward 时 0/0)
        # 对角线强制为 0
        diag_mask = torch.eye(N, dtype=torch.bool, device=points.device)
        dist = dist.clamp(min=_EPS).masked_fill(diag_mask, 0.0)
        return dist

    # ==================== Rips filtration ====================

    def soft_rips_filtration(
        self,
        dist_matrix: torch.Tensor,
    ) -> List[PersistenceDiagram]:
        """
        可微 Rips filtration

        H0 (连通分量):
          - Kruskal MST, 每条 MST 边合并两个分量
          - birth = 0, death = MST 边权 (可微)

        H1 (环路) — 标准 Rips 语义:
          一条非 MST 边 e=(u,v) 在阈值 d(u,v) 处加入, 创建环路
            → birth = d(u,v)
          该环路被一个 2-simplex (三角形) 填充而消亡
          三角形 (u,v,x) 进入复形的阈值 = max(d(u,v), d(u,x), d(v,x))
          环路被最早填充的三角形 kill:
            → death = min_x max(d(u,x), d(v,x), d(u,v))
                    = min_x max(d(u,x), d(v,x))   (因为 d(u,v) ≤ 这两个)
                    ... 实际不一定, 所以保留三项 max

          只保留 death > birth (有正 persistence) 的特征
          等边三角形: birth = death → persistence = 0 → 正确过滤
        """
        N = dist_matrix.shape[0]
        device = dist_matrix.device

        if N <= 1:
            return self._empty_diagrams(device)

        # 提取上三角边
        triu_idx = torch.triu_indices(N, N, offset=1, device=device)
        edge_weights = dist_matrix[triu_idx[0], triu_idx[1]]

        # 排序
        sorted_weights, sort_idx = torch.sort(edge_weights)
        sorted_src = triu_idx[0][sort_idx]
        sorted_dst = triu_idx[1][sort_idx]

        # --- Kruskal MST ---
        parent = list(range(N))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        mst_edges: List[Tuple[int, int, torch.Tensor]] = []
        non_mst_edges: List[Tuple[int, int, torch.Tensor]] = []

        for i in range(len(sorted_weights)):
            u, v = int(sorted_src[i]), int(sorted_dst[i])
            ru, rv = find(u), find(v)
            if ru != rv:
                parent[ru] = rv
                mst_edges.append((u, v, sorted_weights[i]))
            else:
                non_mst_edges.append((u, v, sorted_weights[i]))

        # --- H0 diagram ---
        if mst_edges:
            h0_births = torch.zeros(len(mst_edges), device=device)
            h0_deaths = torch.stack([e[2] for e in mst_edges])
        else:
            h0_births = torch.zeros(1, device=device)
            h0_deaths = torch.zeros(1, device=device)

        diagrams: List[PersistenceDiagram] = [(h0_births, h0_deaths)]

        # --- H1 diagram (向量化) ---
        if self.max_homology_dim >= 1:
            if non_mst_edges:
                # 限制 H1 候选数量: 取 persistence 最可能大的前 max_h1 条
                max_h1 = min(len(non_mst_edges), N, 64)  # 上限 64 条, 足够拓扑约束
                edges_h1 = non_mst_edges[:max_h1]
                M = len(edges_h1)

                # 批量提取 u, v 索引和边权
                us = torch.tensor([e[0] for e in edges_h1], dtype=torch.long, device=device)
                vs = torch.tensor([e[1] for e in edges_h1], dtype=torch.long, device=device)
                ws = torch.stack([e[2] for e in edges_h1])  # [M]

                # 批量索引距离矩阵: [M, N]
                d_ux = dist_matrix[us]  # [M, N]
                d_vx = dist_matrix[vs]  # [M, N]

                # 三角形阈值: max(d(u,x), d(v,x), w) — 向量化
                triangle_thresh = torch.maximum(d_ux, d_vx)  # [M, N]
                triangle_thresh = torch.maximum(triangle_thresh, ws.unsqueeze(1).expand_as(triangle_thresh))

                # 遮蔽 x=u, x=v (不构成有效三角形)
                mask_self = torch.zeros(M, N, device=device)
                mask_self[torch.arange(M, device=device), us] = float('inf')
                mask_self[torch.arange(M, device=device), vs] = float('inf')
                triangle_thresh = triangle_thresh + mask_self

                # death = 每条边的最小三角形阈值
                deaths = triangle_thresh.min(dim=1).values  # [M]
                births = ws  # [M]

                # 过滤 persistence > 0
                valid = deaths > births + _EPS
                if valid.any():
                    h1_births = births[valid]
                    h1_deaths = deaths[valid]
                else:
                    h1_births = torch.zeros(1, device=device)
                    h1_deaths = torch.zeros(1, device=device)
            else:
                h1_births = torch.zeros(1, device=device)
                h1_deaths = torch.zeros(1, device=device)

            diagrams.append((h1_births, h1_deaths))

        return diagrams

    # ==================== Rips via giotto-tda ====================

    def _rips_giotto(self, points: torch.Tensor) -> List[PersistenceDiagram]:
        """
        giotto-tda 精确 Rips + 梯度穿透

        giotto 本身不可微, 我们这样融合:
          - 用 giotto 计算精确的拓扑结构 (哪些 birth/death 值)
          - 从可微距离矩阵中索引对应的边权, 使梯度可回传

        H0 特殊处理:
          - Rips H0 的 birth 恒为 0 (每个点自身就是 0-simplex)
          - 只对 death 做梯度穿透 (匹配到可微距离矩阵的边权)

        H1+:
          - birth 和 death 都从可微距离矩阵匹配
        """
        import numpy as np
        from gtda.homology import VietorisRipsPersistence

        device = points.device
        N = points.shape[0]

        pts_np = points.detach().cpu().numpy().reshape(1, N, -1)
        vr = VietorisRipsPersistence(
            homology_dimensions=list(range(self.max_homology_dim + 1)),
            n_jobs=1,
        )
        pd_np = vr.fit_transform(pts_np)[0]  # [[birth, death, dim], ...]

        # 可微距离矩阵 + 边权向量
        dist_mat = self.compute_distance_matrix(points)
        triu = torch.triu_indices(N, N, offset=1, device=device)
        edge_vals = dist_mat[triu[0], triu[1]]       # [E] 可微
        edge_detached = edge_vals.detach()            # [E] 用于匹配

        def _match_to_edge(val: float) -> torch.Tensor:
            """在可微边权中找最近匹配, 返回可微 tensor"""
            idx = (edge_detached - val).abs().argmin()
            return edge_vals[idx]

        diagrams = []
        for dim in range(self.max_homology_dim + 1):
            mask = pd_np[:, 2] == dim
            pairs = pd_np[mask]
            finite_mask = np.isfinite(pairs[:, 1])
            pairs = pairs[finite_mask]

            if len(pairs) > 0:
                if dim == 0:
                    # H0: birth 恒为 0, 只对 death 做梯度穿透
                    births = torch.zeros(len(pairs), device=device)
                    deaths = torch.stack([_match_to_edge(d) for d in pairs[:, 1]])
                else:
                    # H1+: birth 和 death 都做梯度穿透
                    births = torch.stack([_match_to_edge(b) for b in pairs[:, 0]])
                    deaths = torch.stack([_match_to_edge(d) for d in pairs[:, 1]])
            else:
                births = torch.zeros(1, device=device)
                deaths = torch.zeros(1, device=device)

            diagrams.append((births, deaths))

        return diagrams

    # ==================== Cubical filtration ====================

    def cubical_filtration(
        self,
        scalar_field: torch.Tensor,            # [H, W]
    ) -> List[PersistenceDiagram]:
        """
        Cubical filtration (用于 2D 特征图)

        后端: gudhi (精确) → 纯 PyTorch (近似)
        """
        if self._has_gudhi:
            return self._cubical_gudhi(scalar_field)
        return self._cubical_pytorch(scalar_field)

    def _cubical_pytorch(self, scalar_field: torch.Tensor) -> List[PersistenceDiagram]:
        """
        纯 PyTorch cubical 近似

        H0: sublevel set Union-Find + elder rule
          - 按函数值从小到大激活像素
          - 合并相邻分量时, 较年轻 (birth 值更大) 的分量 die
          - birth = 分量根的函数值, death = 合并时的函数值

        H1: 非树边产生环路 (简版)
          - 在 sublevel set 生长树中, 非树边 = 环路
          - birth = 该边的函数值 (环路出现时的阈值)
          - death = 环路被填充时的阈值 (近似为局部极大值)
        """
        device = scalar_field.device
        H, W = scalar_field.shape
        num_pixels = H * W

        flat = scalar_field.reshape(-1)
        sorted_vals, sorted_idx = torch.sort(flat)

        parent = list(range(num_pixels))
        rank = [0] * num_pixels
        birth_val = [None] * num_pixels   # 每个根的 birth (函数值 tensor)
        active = [False] * num_pixels

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def neighbors(idx):
            r, c = idx // W, idx % W
            nbs = []
            if r > 0: nbs.append(idx - W)
            if r < H - 1: nbs.append(idx + W)
            if c > 0: nbs.append(idx - 1)
            if c < W - 1: nbs.append(idx + 1)
            return nbs

        h0_births = []
        h0_deaths = []
        # H1: 非树边 (两端点已在同一分量)
        h1_births = []

        for i in range(len(sorted_vals)):
            p = int(sorted_idx[i])
            active[p] = True
            parent[p] = p
            birth_val[p] = sorted_vals[i]

            for nb in neighbors(p):
                if not active[nb]:
                    continue
                rp, rnb = find(p), find(nb)
                if rp != rnb:
                    # 树边: 合并 (elder rule — birth 较大者 die)
                    bp = birth_val[rp]
                    bnb = birth_val[rnb]
                    if bp.item() >= bnb.item():
                        # rp 更年轻, die
                        h0_births.append(bp)
                        h0_deaths.append(sorted_vals[i])
                        # union by rank
                        if rank[rnb] < rank[rp]:
                            rank[rnb] = rank[rp]
                        parent[rp] = rnb
                    else:
                        h0_births.append(bnb)
                        h0_deaths.append(sorted_vals[i])
                        if rank[rp] < rank[rnb]:
                            rank[rp] = rank[rnb]
                        parent[rnb] = rp
                else:
                    # 非树边: 产生 H1 环路
                    # birth = 当前函数值 (环路出现的阈值)
                    if self.max_homology_dim >= 1:
                        h1_births.append(sorted_vals[i])

        # H0 diagram
        if h0_births:
            h0_b = torch.stack(h0_births)
            h0_d = torch.stack(h0_deaths)
        else:
            h0_b = torch.zeros(1, device=device)
            h0_d = torch.zeros(1, device=device)

        diagrams: List[PersistenceDiagram] = [(h0_b, h0_d)]

        # H1 diagram (简版)
        if self.max_homology_dim >= 1:
            if h1_births:
                h1_b = torch.stack(h1_births)
                # death 近似: 取全局最大值 (所有环路最终在最高阈值被填充)
                # 更精确需要跟踪每个环路何时被填充, 简版先用 max
                global_max = flat.max()
                h1_d = global_max.expand_as(h1_b).clone()
                # 过滤 persistence 太小的
                persistence = h1_d - h1_b
                valid = persistence > _EPS
                if valid.any():
                    h1_b = h1_b[valid]
                    h1_d = h1_d[valid]
                else:
                    h1_b = torch.zeros(1, device=device)
                    h1_d = torch.zeros(1, device=device)
            else:
                h1_b = torch.zeros(1, device=device)
                h1_d = torch.zeros(1, device=device)

            diagrams.append((h1_b, h1_d))

        return diagrams

    def _cubical_gudhi(self, scalar_field: torch.Tensor) -> List[PersistenceDiagram]:
        """gudhi CubicalComplex 精确计算 + 值索引梯度穿透"""
        import gudhi
        import numpy as np

        device = scalar_field.device
        data = scalar_field.detach().cpu().numpy()
        flat_torch = scalar_field.reshape(-1)

        cc = gudhi.CubicalComplex(
            dimensions=list(data.shape),
            top_dimensional_cells=data.flatten(),
        )
        cc.persistence()

        diagrams = []
        for dim in range(self.max_homology_dim + 1):
            pairs = cc.persistence_intervals_in_dimension(dim)
            if len(pairs) > 0:
                pairs = np.array(pairs)
                finite_mask = np.isfinite(pairs[:, 1])
                pairs = pairs[finite_mask]
                if len(pairs) > 0:
                    # 梯度穿透: 在可微 tensor 中找最近匹配值
                    flat_detached = flat_torch.detach()
                    birth_tensors = []
                    death_tensors = []
                    for b_val, d_val in zip(pairs[:, 0], pairs[:, 1]):
                        b_idx = (flat_detached - b_val).abs().argmin()
                        d_idx = (flat_detached - d_val).abs().argmin()
                        birth_tensors.append(flat_torch[b_idx])
                        death_tensors.append(flat_torch[d_idx])
                    births = torch.stack(birth_tensors)
                    deaths = torch.stack(death_tensors)
                else:
                    births = torch.zeros(1, device=device)
                    deaths = torch.zeros(1, device=device)
            else:
                births = torch.zeros(1, device=device)
                deaths = torch.zeros(1, device=device)

            diagrams.append((births, deaths))

        return diagrams

    # ==================== 入口 ====================

    def _empty_diagrams(self, device: torch.device) -> List[PersistenceDiagram]:
        """空 diagrams (点数不足时)"""
        return [
            (torch.zeros(1, device=device), torch.zeros(1, device=device))
            for _ in range(self.max_homology_dim + 1)
        ]

    def forward(
        self,
        points: torch.Tensor,                 # [N, D] (rips) 或 [H, W] (cubical)
    ) -> List[PersistenceDiagram]:
        """
        根据 filtration_type 分派计算

        Rips 后端优先级: giotto-tda (精确) → 纯 PyTorch (近似)
        Cubical 后端优先级: gudhi (精确) → 纯 PyTorch (近似)
        """
        if self.filtration_type == "rips":
            if self._has_giotto and points.shape[0] >= 2:
                return self._rips_giotto(points)
            dist_mat = self.compute_distance_matrix(points)
            return self.soft_rips_filtration(dist_mat)
        elif self.filtration_type == "cubical":
            return self.cubical_filtration(points)
        else:
            raise ValueError(f"Unknown filtration type: {self.filtration_type}")


# ============================================================
# 拓扑分支 (持有 DifferentiablePH)
# ============================================================

class TopologicalBranch(nn.Module):
    """
    拓扑分支: 计算嵌入空间和特征空间的持久同调

    持有两个 DifferentiablePH 实例:
      - rips_ph: 用于嵌入空间 (高维点云)
      - cubical_ph: 用于特征空间 (2D 特征图)
    """

    def __init__(
        self,
        embed_dim: int = 768,
        max_homology_dim: int = 1,
        max_points: int = 256,
        persistence_threshold: float = 0.01,
        vectorization_method: str = "landscape",
        vectorization_dim: int = 64,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_homology_dim = max_homology_dim
        self.max_points = max_points
        self.persistence_threshold = persistence_threshold
        self.vectorization_method = vectorization_method
        self.vectorization_dim = vectorization_dim

        self.rips_ph = DifferentiablePH(
            max_homology_dim=max_homology_dim,
            max_points=max_points,
            filtration_type="rips",
        )
        self.cubical_ph = DifferentiablePH(
            max_homology_dim=max_homology_dim,
            max_points=max_points,
            filtration_type="cubical",
        )

    # ============ 嵌入空间 PH ============

    def compute_embedding_diagrams(
        self,
        embeddings: torch.Tensor,              # [B, D]
    ) -> List[PersistenceDiagram]:
        """
        计算嵌入点云的持久图
        将 B 个 D 维向量视为 D 维空间中的 B 个点
        如果 B > max_points, 随机子采样
        """
        B = embeddings.shape[0]
        if B > self.max_points:
            indices = torch.randperm(B, device=embeddings.device)[:self.max_points]
            embeddings = embeddings[indices]
        return self.rips_ph(embeddings)

    # ============ 特征空间 PH ============

    # cubical PH 的特征图最大尺寸 (超过则下采样)
    # 64x64 = 4096 像素, Union-Find 可在 ~10ms 内完成
    _CUBICAL_MAX_SIZE: int = 64

    def compute_feature_diagrams(
        self,
        feature_map: torch.Tensor,             # [B, C, H, W]
    ) -> List[PersistenceDiagram]:
        """
        从 2D 特征图计算持久图
        先对每个样本做 channel-wise L2 norm，再在 batch 维度上取均值

        论文里的 feature-space PH 是对中间 feature map 的 cubical persistence；
        这里输出 batch-level 近似，用于稳定训练和低开销对齐。
        """
        scalar_field = feature_map.norm(dim=1)   # [B, H, W]
        scalar_field = scalar_field.mean(dim=0)   # [H, W]
        scalar_field = self._normalize_scalar_field(scalar_field)

        H, W = scalar_field.shape
        max_sz = self._CUBICAL_MAX_SIZE
        if H > max_sz or W > max_sz:
            scale = max_sz / max(H, W)
            new_h, new_w = max(1, int(H * scale)), max(1, int(W * scale))
            scalar_field = torch.nn.functional.interpolate(
                scalar_field.unsqueeze(0).unsqueeze(0),
                size=(new_h, new_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0)

        return self.cubical_ph(scalar_field)

    def _normalize_scalar_field(self, scalar_field: torch.Tensor) -> torch.Tensor:
        """将特征场归一化到 [0, 1] 以匹配论文中的统一滤波范围。"""
        min_val = scalar_field.min()
        max_val = scalar_field.max()
        denom = (max_val - min_val).clamp(min=_EPS)
        return (scalar_field - min_val) / denom

    # ============ Betti 数 ============

    def compute_betti_numbers(
        self,
        diagrams: List[PersistenceDiagram],
        threshold: Optional[float] = None,
        soft: bool = True,
        temperature: float = 0.01,
    ) -> torch.Tensor:
        """
        从持久图计算 Betti 数

        soft=True (默认): 可微 soft counting, 梯度可回传到 PH
          count = Σ sigmoid((persistence - threshold) / temperature)
        soft=False: 硬计数 (不可微, 仅用于 logging/metrics)
          count = #{features : persistence > threshold}

        Betti_k = count of features in dim k
        """
        threshold = threshold or self.persistence_threshold
        device = diagrams[0][0].device

        betti = []
        for births, deaths in diagrams:
            persistence = (deaths - births).abs()
            if soft:
                # 可微 soft counting: sigmoid 近似阶跃函数
                count = torch.sigmoid((persistence - threshold) / temperature).sum()
            else:
                count = (persistence > threshold).sum().float()
            betti.append(count)

        return torch.stack(betti).to(device)

    # ============ 多尺度特征拓扑 ============

    def _aggregate_multi_scale_diagrams(
        self,
        multi_scale_diagrams: List[List[PersistenceDiagram]],
        weights: Optional[List[float]] = None,
    ) -> List[PersistenceDiagram]:
        """
        聚合多尺度特征图的持久图

        策略: 按尺度加权拼接 birth/death 值
          - P3 (高分辨率) 权重最大, 拓扑信息最丰富
          - P5 (低分辨率) 权重最小

        Args:
            multi_scale_diagrams: [[diagrams_scale0], [diagrams_scale1], ...]
                每个 scale 的 diagrams 是 List[PersistenceDiagram] (按 homology dim)
            weights: 每个尺度的权重, 默认 [1.0, 0.5, 0.25, ...] (递减)

        Returns:
            aggregated: List[PersistenceDiagram] (按 homology dim)
        """
        if not multi_scale_diagrams:
            return []

        n_scales = len(multi_scale_diagrams)
        if weights is None:
            # 高分辨率 (P3) 权重大, 低分辨率 (P5) 权重小
            weights = [1.0 / (2 ** i) for i in range(n_scales)]
        # 归一化
        w_sum = sum(weights)
        weights = [w / w_sum for w in weights]

        # 确定最大 homology dim
        max_dims = max(len(diags) for diags in multi_scale_diagrams)

        aggregated: List[PersistenceDiagram] = []
        for dim in range(max_dims):
            all_births = []
            all_deaths = []
            for scale_idx, diags in enumerate(multi_scale_diagrams):
                if dim < len(diags):
                    births, deaths = diags[dim]
                    w = weights[scale_idx]
                    # 加权: 缩放 persistence 值
                    all_births.append(births * w)
                    all_deaths.append(deaths * w)

            if all_births:
                agg_births = torch.cat(all_births, dim=0)
                agg_deaths = torch.cat(all_deaths, dim=0)
            else:
                device = multi_scale_diagrams[0][0][0].device
                agg_births = torch.zeros(1, device=device)
                agg_deaths = torch.zeros(1, device=device)

            aggregated.append((agg_births, agg_deaths))

        return aggregated

    # ============ 主前向 ============

    def compute_embedding_diagrams_h0_only(
        self,
        embeddings: torch.Tensor,              # [B, D]
    ) -> List[PersistenceDiagram]:
        """
        仅计算 H0 (连通分量) 的嵌入持久图, 作为低成本参考分支。

        论文中的 fidelity 项比较 predicted / target embeddings 的 H0 与 H1。
        这里保留一个轻量版本用于 warm-up 或调试；主训练路径仍使用完整 PH。
        """
        B = embeddings.shape[0]
        if B > self.max_points:
            indices = torch.randperm(B, device=embeddings.device)[:self.max_points]
            embeddings = embeddings[indices]

        saved_dim = self.rips_ph.max_homology_dim
        self.rips_ph.max_homology_dim = 0
        try:
            diagrams = self.rips_ph(embeddings)
        finally:
            self.rips_ph.max_homology_dim = saved_dim

        while len(diagrams) <= saved_dim:
            diagrams.append((
                torch.zeros(1, device=embeddings.device),
                torch.zeros(1, device=embeddings.device),
            ))
        return diagrams

    def forward(
        self,
        pred_embeddings: torch.Tensor,         # [B, D]
        target_embeddings: torch.Tensor,       # [B, D]
        feature_maps: Optional[List[torch.Tensor]] = None,
    ) -> Dict:
        """
        计算完整的拓扑信息

        统一返回:
          - pred/target diagrams
          - feature diagrams
          - betti 曲线和总持久性
        """
        pred_diagrams = self.compute_embedding_diagrams(pred_embeddings)
        target_diagrams = self.compute_embedding_diagrams(target_embeddings)

        feature_diagrams = None
        if feature_maps is not None and len(feature_maps) > 0:
            multi_scale = []
            for fm in feature_maps:
                try:
                    diags = self.compute_feature_diagrams(fm)
                    multi_scale.append(diags)
                except Exception:
                    continue
            if multi_scale:
                feature_diagrams = self._aggregate_multi_scale_diagrams(multi_scale)

        pred_betti = self.compute_betti_numbers(pred_diagrams)
        target_betti = self.compute_betti_numbers(target_diagrams)
        pred_tp = torch.stack([torch.clamp(d[1] - d[0], min=0).sum() for d in pred_diagrams])
        target_tp = torch.stack([torch.clamp(d[1] - d[0], min=0).sum() for d in target_diagrams])

        return {
            "pred_diagrams": pred_diagrams,
            "target_diagrams": target_diagrams,
            "feature_diagrams": feature_diagrams,
            "pred_betti": pred_betti,
            "target_betti": target_betti,
            "pred_total_persistence": pred_tp,
            "target_total_persistence": target_tp,
        }
