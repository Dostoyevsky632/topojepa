"""
Encoder 模块
============
包含两个编码器:
  - VisualEncoder (X-Encoder): 视觉输入 → 视觉嵌入 S_V
    借鉴 jepa/src/models/vision_transformer.py 的 VisionTransformer
    实际使用中可选: 冻结的 ViT (来自 V-JEPA 2) 或 YOLO backbone 特征

  - TextEncoder (Y-Encoder): 文本目标 → 目标嵌入 S_Y
    用于将类别标签/描述文本编码到共享嵌入空间
    参考 VL-JEPA 的 EmbeddingGemma, 实际使用 sentence-transformers
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple


# ============================================================
# ViT 模式所需的子模块
# ============================================================

class PatchEmbed(nn.Module):
    """将图像拆分为 patch 并嵌入 (参考 jepa/src/models/utils/patch_embed.py)"""

    def __init__(self, img_size: int = 640, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            [B, N, D]  N = (H/p)*(W/p)
        """
        x = self.proj(x)           # [B, D, H/p, W/p]
        x = x.flatten(2)           # [B, D, N]
        x = x.transpose(1, 2)     # [B, N, D]
        return x


class TransformerBlock(nn.Module):
    """标准 Pre-LN Transformer Block (参考 jepa Block)"""

    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-LN self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, key_padding_mask=mask)
        x = x + attn_out
        # Pre-LN FFN
        x = x + self.mlp(self.norm2(x))
        return x


# ============================================================
# YOLO backbone 特征 → 嵌入序列的适配器
# ============================================================

class YOLOFeatureAdapter(nn.Module):
    """
    将 YOLO backbone 的多尺度特征图 (P3/P4/P5) 投影到统一嵌入空间
    1. 各尺度特征图分别 1×1 conv → embed_dim
    2. flatten + concat → 嵌入序列 [B, N_total, D]
    """

    def __init__(self, in_channels: List[int], embed_dim: int = 768):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, embed_dim, 1),
                nn.BatchNorm2d(embed_dim),
                nn.GELU(),
            )
            for ch in in_channels
        ])

    def forward(self, feature_maps: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            feature_maps: [P3, P4, P5], 各 [B, C_i, H_i, W_i]
        Returns:
            [B, N_total, D]  N_total = sum(H_i * W_i)
        """
        projected = []
        for proj, fm in zip(self.projections, feature_maps):
            x = proj(fm)                    # [B, D, H_i, W_i]
            B, D, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)  # [B, H_i*W_i, D]
            projected.append(x)
        return torch.cat(projected, dim=1)  # [B, N_total, D]


# ============================================================
# VisualEncoder
# ============================================================

class VisualEncoder(nn.Module):
    """
    视觉编码器: 将图像/视频编码为视觉嵌入序列

    设计选择:
      - 模式 A (ViT): 直接用预训练 ViT, 类似原始 JEPA
      - 模式 B (YOLO backbone): 复用 ultralytics YOLO 的 backbone 特征
        从多尺度特征图 (P3/P4/P5) 提取, 然后 flatten + project 到嵌入空间

    与原始 JEPA 的区别:
      - 原始 JEPA 用 patch masking 做自监督预训练
      - 我们的场景是有监督的 (有 YOLO 标注), 所以 encoder 可以冻结或微调
    """

    # YOLO v8/v11 backbone 中 P3/P4/P5 对应的层索引
    # (基于 yolov8.yaml: backbone 10 层, P3=layer4, P4=layer6, P5=layer9)
    _YOLO_FEATURE_LAYERS = [4, 6, 9]

    def __init__(
        self,
        backbone_type: str = "vit",          # "vit" | "yolo_backbone"
        pretrained_path: Optional[str] = None,
        embed_dim: int = 768,                 # 输出嵌入维度
        patch_size: int = 16,
        img_size: int = 640,
        freeze: bool = True,                  # 是否冻结 backbone
        yolo_feature_layers: List[int] = None,  # YOLO 模式下提取哪些层的特征
    ):
        """初始化视觉编码器"""
        super().__init__()
        self.backbone_type = backbone_type
        self.embed_dim = embed_dim
        self.freeze = freeze
        self.yolo_feature_layers = yolo_feature_layers or self._YOLO_FEATURE_LAYERS

        if backbone_type == "vit":
            self._build_vit(img_size, patch_size, embed_dim)
        elif backbone_type == "yolo_backbone":
            self._build_yolo_backbone(pretrained_path, embed_dim)
        else:
            raise ValueError(f"Unknown backbone_type: {backbone_type}")

        # 冻结
        if freeze:
            self._freeze_backbone()

    # ---------- ViT 模式 ----------

    def _build_vit(self, img_size: int, patch_size: int, embed_dim: int):
        """构建 ViT backbone (参考 jepa VisionTransformer)"""
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        num_patches = self.patch_embed.num_patches

        # 可学习位置编码 (1, N, D)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # num_heads 自适应: 优先 12, 回退到能整除的最大值
        num_heads = 12
        while embed_dim % num_heads != 0 and num_heads > 1:
            num_heads -= 1

        # 6 层 Transformer (轻量, 因为可能在 frozen 模式使用)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads=num_heads, mlp_ratio=4.0)
            for _ in range(6)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # ViT 模式没有多尺度特征图
        self._yolo_model = None
        self._feature_adapter = None

    def _vit_forward(self, x: torch.Tensor) -> torch.Tensor:
        """ViT 前向: [B, C, H, W] → [B, N, D]"""
        x = self.patch_embed(x)                     # [B, N, D]
        # 位置编码插值 (如果图像尺寸与训练时不同)
        pos = self._interpolate_pos_embed(x)
        x = x + pos
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

    def _interpolate_pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        """
        位置编码插值: 适配不同分辨率输入
        参考 jepa/src/models/vision_transformer.py:interpolate_pos_encoding
        """
        N = x.shape[1]
        pos = self.pos_embed
        if pos.shape[1] == N:
            return pos
        # 需要 2D 插值
        old_N = pos.shape[1]
        old_size = int(math.sqrt(old_N))
        new_size = int(math.sqrt(N))
        pos = pos.reshape(1, old_size, old_size, -1).permute(0, 3, 1, 2)
        pos = F.interpolate(pos, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos = pos.permute(0, 2, 3, 1).reshape(1, new_size * new_size, -1)
        return pos

    # ---------- YOLO backbone 模式 ----------

    def _build_yolo_backbone(self, pretrained_path: Optional[str], embed_dim: int):
        """构建 YOLO backbone + 特征适配器"""
        self._yolo_model = None
        self._backbone_layers = None

        if pretrained_path is not None:
            self._load_yolo_backbone(pretrained_path)

        # 默认通道数 (YOLOv8l/v11l), 如果 backbone 已加载会被覆盖
        default_channels = [256, 512, 1024]
        in_channels = getattr(self, '_feature_channels', default_channels)
        self._feature_adapter = YOLOFeatureAdapter(in_channels, embed_dim)

        # ViT 组件置空
        self.patch_embed = None
        self.pos_embed = None
        self.blocks = None
        self.norm = None

    def _load_yolo_backbone(self, model_path: str):
        """
        加载 ultralytics YOLO 模型, 提取 backbone 层

        ultralytics model.model 是一个 Sequential, 前 10 层为 backbone:
          - layer 0: Conv (stem)
          - layer 1-9: backbone blocks
          - layer 4: 产出 P3/8
          - layer 6: 产出 P4/16
          - layer 9: 产出 P5/32 (含 SPPF)
        """
        try:
            from ultralytics import YOLO
            yolo = YOLO(model_path)
            model = yolo.model.model  # nn.Sequential of layers

            # 确定 backbone 范围 (前 10 层)
            backbone_end = max(self.yolo_feature_layers) + 1
            self._backbone_layers = nn.ModuleList([model[i] for i in range(backbone_end)])

            # 探测各 feature layer 的输出通道数
            # 优先用一次 dummy forward 获取真实输出通道，避免静态遍历模块时误判
            self._feature_channels = self._infer_feature_channels_via_forward()

            # 更新适配器
            self._feature_adapter = YOLOFeatureAdapter(self._feature_channels, self.embed_dim)

        except ImportError:
            raise ImportError("ultralytics is required for YOLO backbone mode. pip install ultralytics")
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO backbone from {model_path}: {e}")

    def _infer_feature_channels_via_forward(self, probe_size: int = 256) -> List[int]:
        """
        通过一次无梯度前向获取真实 feature channels

        比静态遍历 module.modules() 更稳健，能够正确处理含分支/残差的复合模块。
        """
        if self._backbone_layers is None:
            return []

        # 用 backbone 参数设备/精度构造 probe，避免 device mismatch
        first_param = next(self._backbone_layers.parameters(), None)
        device = first_param.device if first_param is not None else torch.device("cpu")
        dtype = first_param.dtype if first_param is not None else torch.float32

        # 临时切 eval，避免 BN/dropout 引入随机性；结束后恢复
        was_training = self._backbone_layers.training
        self._backbone_layers.eval()
        try:
            with torch.no_grad():
                probe = torch.zeros(1, 3, probe_size, probe_size, device=device, dtype=dtype)
                features = self._yolo_extract_features(probe)
            if len(features) != len(self.yolo_feature_layers):
                raise RuntimeError(
                    f"Expected {len(self.yolo_feature_layers)} features, got {len(features)} "
                    f"for layers {self.yolo_feature_layers}"
                )
            return [int(f.shape[1]) for f in features]
        finally:
            self._backbone_layers.train(was_training)

    @staticmethod
    def _infer_out_channels(module: nn.Module) -> int:
        """从模块推断输出通道数 (遍历子模块找最后一个 Conv2d/BatchNorm2d)"""
        out_ch = 256  # fallback
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                out_ch = m.out_channels
            elif isinstance(m, nn.BatchNorm2d):
                out_ch = m.num_features
        return out_ch

    def _yolo_extract_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        通过 YOLO backbone 前向, 收集指定层的特征图

        解冻层在 fp32 下运行 (禁用 autocast), 避免 BN + fp16 的数值溢出。

        Returns:
            [P3, P4, P5] 多尺度特征图
        """
        features = []
        if self._backbone_layers is not None:
            total = len(self._backbone_layers)
            # 判断哪些层被解冻 (有 requires_grad=True 的参数)
            unfrozen_start = total  # 默认无解冻层
            for i in range(total):
                if any(p.requires_grad for p in self._backbone_layers[i].parameters()):
                    unfrozen_start = i
                    break

            y = []  # 各层输出缓存 (ultralytics 风格)
            for i, layer in enumerate(self._backbone_layers):
                # ultralytics 层有 .f 属性指定输入来源
                if hasattr(layer, 'f') and layer.f != -1:
                    if isinstance(layer.f, int):
                        x = y[layer.f]
                    else:
                        x = [y[j] if j != -1 else x for j in layer.f]

                # 解冻层: 在 fp32 下运行, 避免 AMP fp16 导致 BN 数值溢出
                if i >= unfrozen_start:
                    _dev = x.device if isinstance(x, torch.Tensor) else (
                        x[0].device if isinstance(x, list) and len(x) > 0 and isinstance(x[0], torch.Tensor) else torch.device("cpu"))
                    _amp_dev = "cuda" if _dev.type == "cuda" else "cpu"
                    with torch.amp.autocast(_amp_dev, enabled=False):
                        if isinstance(x, torch.Tensor):
                            x = layer(x.float())
                        elif isinstance(x, list):
                            x = layer([t.float() if isinstance(t, torch.Tensor) else t for t in x])
                        else:
                            x = layer(x)
                else:
                    x = layer(x)

                y.append(x)
                if i in self.yolo_feature_layers:
                    features.append(x)
        return features

    def _yolo_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        YOLO backbone 前向: [B, C, H, W] → ([B, N, D], [P3, P4, P5])

        当 backbone 有解冻层时, feature_maps 可能是 fp32 (来自禁用 autocast 的解冻层)。
        为避免 adapter 在 autocast fp16 下接收 fp32 大值导致溢出,
        adapter 也在 fp32 下运行。
        """
        feature_maps = self._yolo_extract_features(x)
        if not feature_maps:
            raise RuntimeError("YOLO backbone did not produce feature maps. Check model loading.")
        # 特征图 → 嵌入序列
        # 如果特征图是 fp32 (解冻层输出), adapter 也在 fp32 下运行
        any_fp32 = any(f.dtype == torch.float32 for f in feature_maps)
        if any_fp32:
            _amp_dev = "cuda" if feature_maps[0].device.type == "cuda" else "cpu"
            with torch.amp.autocast(_amp_dev, enabled=False):
                S_V = self._feature_adapter(feature_maps)   # [B, N_total, D]
        else:
            S_V = self._feature_adapter(feature_maps)   # [B, N_total, D]
        return S_V, feature_maps

    # ---------- 冻结 ----------

    def _freeze_backbone(self):
        """冻结 backbone 参数 (保留 adapter 可训练)"""
        if self.backbone_type == "vit":
            # 冻结 ViT
            if self.patch_embed is not None:
                for p in self.patch_embed.parameters():
                    p.requires_grad = False
            if self.pos_embed is not None:
                self.pos_embed.requires_grad = False
            if self.blocks is not None:
                for p in self.blocks.parameters():
                    p.requires_grad = False
            if self.norm is not None:
                for p in self.norm.parameters():
                    p.requires_grad = False
        elif self.backbone_type == "yolo_backbone":
            # 冻结 YOLO backbone, 保留 adapter 可训练
            if self._backbone_layers is not None:
                for p in self._backbone_layers.parameters():
                    p.requires_grad = False

    def partial_unfreeze(self, n_layers: int = 3):
        """
        解冻 backbone 最后 n_layers 层 (Stage 2 微调用)

        YOLO backbone: _backbone_layers 是 [layer0, ..., layer9]
          - 解冻最后 3 层 → layer7, layer8, layer9 (含 SPPF)
          - 这些层产出 P4/P5 特征, 对检测最关键

        ViT: 解冻最后 n_layers 个 Transformer block + norm

        同时将解冻层中的 BatchNorm 强制转为 fp32, 避免 AMP fp16 下的数值溢出。

        Returns:
            n_unfrozen: 实际解冻的参数数量
        """
        n_unfrozen = 0
        if self.backbone_type == "yolo_backbone" and self._backbone_layers is not None:
            total = len(self._backbone_layers)
            start = max(0, total - n_layers)
            for i in range(start, total):
                for p in self._backbone_layers[i].parameters():
                    if not p.requires_grad:
                        p.requires_grad = True
                        n_unfrozen += 1
                # BN 层强制 fp32 (AMP 安全)
                self._force_bn_fp32(self._backbone_layers[i])
        elif self.backbone_type == "vit" and self.blocks is not None:
            total = len(self.blocks)
            start = max(0, total - n_layers)
            for i in range(start, total):
                for p in self.blocks[i].parameters():
                    if not p.requires_grad:
                        p.requires_grad = True
                        n_unfrozen += 1
            # 同时解冻 norm (与最后几层配合)
            if self.norm is not None:
                for p in self.norm.parameters():
                    if not p.requires_grad:
                        p.requires_grad = True
                        n_unfrozen += 1
        return n_unfrozen

    @staticmethod
    def _force_bn_fp32(module: nn.Module):
        """将模块内所有 BatchNorm 层强制转为 fp32, 防止 AMP 下数值不稳定"""
        for m in module.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.SyncBatchNorm)):
                m.float()

    # ---------- 公开接口 ----------

    def forward(
        self,
        x: torch.Tensor,                      # [B, C, H, W] 图像输入
        masks: Optional[List[torch.Tensor]] = None,  # 可选的 mask 索引 (JEPA 预训练用)
    ) -> torch.Tensor:
        """
        前向传播

        Returns:
            S_V: [B, N, D] 视觉嵌入序列, N=patch数量, D=embed_dim
        """
        if self.backbone_type == "vit":
            return self._vit_forward(x)
        elif self.backbone_type == "yolo_backbone":
            S_V, _ = self._yolo_forward(x)
            return S_V
        else:
            raise ValueError(f"Unknown backbone_type: {self.backbone_type}")

    def extract_multiscale_features(
        self,
        x: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        提取多尺度特征图 (仅 YOLO backbone 模式)
        用于后续的拓扑分支和检测头

        Returns:
            features: [P3, P4, P5] 多尺度特征图列表
        """
        if self.backbone_type != "yolo_backbone":
            raise RuntimeError("extract_multiscale_features is only available in YOLO backbone mode")
        return self._yolo_extract_features(x)

    def forward_with_features(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        前向传播, 同时返回多尺度特征图 (供 TopoJEPA.forward_visual 调用)

        Returns:
            S_V: [B, N, D]
            feature_maps: [P3, P4, P5] 或 None (ViT 模式)
        """
        if self.backbone_type == "vit":
            return self._vit_forward(x), None
        elif self.backbone_type == "yolo_backbone":
            return self._yolo_forward(x)
        else:
            raise ValueError(f"Unknown backbone_type: {self.backbone_type}")

    @staticmethod
    def from_yolo_model(
        yolo_model_path: str,
        embed_dim: int = 768,
        freeze: bool = True,
    ) -> "VisualEncoder":
        """
        从已有的 ultralytics YOLO 模型构建 VisualEncoder
        提取 backbone 部分, 丢弃检测头
        """
        return VisualEncoder(
            backbone_type="yolo_backbone",
            pretrained_path=yolo_model_path,
            embed_dim=embed_dim,
            freeze=freeze,
        )


# ============================================================
# TextEncoder
# ============================================================

class TextEncoder(nn.Module):
    """
    文本/标签编码器 (Y-Encoder): 将文本目标编码为目标嵌入

    在 VL-JEPA 中这是 EmbeddingGemma-300M
    在我们的场景中, 输入可以是:
      - 类别名称: "Longitudinal Crack", "Pothole" 等
      - 结构化描述: "two parallel longitudinal cracks in the center"
      - bbox 的文本化: 将空间位置信息编码为自然语言

    关键设计: lr_multiplier=0.05 (参考 VL-JEPA 论文的发现)
    """

    # 区域描述映射 (将归一化坐标映射到语义位置词)
    _REGION_MAP = {
        (0, 0): "upper-left",
        (0, 1): "upper-center",
        (0, 2): "upper-right",
        (1, 0): "center-left",
        (1, 1): "center",
        (1, 2): "center-right",
        (2, 0): "lower-left",
        (2, 1): "lower-center",
        (2, 2): "lower-right",
    }

    # 内置 fallback 词表 (不依赖外部库, 保证最小前向可跑)
    _FALLBACK_VOCAB_SIZE = 30522  # BERT-base vocab size
    _FALLBACK_HIDDEN = 256

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",  # sentence-transformers 模型名
        embed_dim: int = 768,                    # 投影后的嵌入维度 (需与 visual 对齐)
        max_length: int = 128,                   # 最大文本长度
        freeze_base: bool = False,               # 是否冻结基础模型
        lr_multiplier: float = 0.05,             # 学习率缩放因子
    ):
        """初始化文本编码器"""
        super().__init__()
        self.model_name = model_name
        self.embed_dim = embed_dim
        self.max_length = max_length
        self.freeze_base = freeze_base
        self.lr_multiplier = lr_multiplier

        # 后端标记: "st" (sentence-transformers) 或 "fallback" (纯 PyTorch)
        self._backend: Optional[str] = None

        # sentence-transformers 后端
        self._st_model = None
        self._st_dim = None

        # fallback 后端 (在 __init__ 时就创建, 避免延迟初始化的 device 问题)
        # 先创建为 None, _ensure_loaded 中按需赋值或构建
        self._fallback_embed = None
        self._fallback_pool = None

        # 投影头: raw_dim → embed_dim
        # 统一在 __init__ 时创建, 确保跟随 model.to(device)
        self.projection = nn.Sequential(
            nn.Linear(self._FALLBACK_HIDDEN, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # 尝试加载 sentence-transformers; 失败则激活 fallback
        self._try_init()

    def _try_init(self):
        """尝试加载 sentence-transformers, 失败则构建 fallback"""
        try:
            from sentence_transformers import SentenceTransformer
            self._st_model = SentenceTransformer(self.model_name)
            self._st_dim = self._st_model.get_sentence_embedding_dimension()
            self._backend = "st"

            # 冻结基础模型
            if self.freeze_base:
                for p in self._st_model.parameters():
                    p.requires_grad = False

            # 重建投影头匹配 st_dim
            if self._st_dim != self.embed_dim:
                self.projection = nn.Sequential(
                    nn.Linear(self._st_dim, self.embed_dim),
                    nn.GELU(),
                    nn.Linear(self.embed_dim, self.embed_dim),
                )
            else:
                self.projection = nn.Identity()

        except (ImportError, Exception):
            # fallback: 纯 PyTorch 字符级编码器
            self._backend = "fallback"
            self._fallback_embed = nn.Embedding(self._FALLBACK_VOCAB_SIZE, self._FALLBACK_HIDDEN)
            nn.init.trunc_normal_(self._fallback_embed.weight, std=0.02)
            self._fallback_pool = nn.Sequential(
                nn.Linear(self._FALLBACK_HIDDEN, self._FALLBACK_HIDDEN),
                nn.GELU(),
            )
            # projection 已在 __init__ 中创建 (FALLBACK_HIDDEN → embed_dim), 无需重建

    def _get_device(self) -> torch.device:
        """安全获取当前模型所在 device"""
        # projection 一定存在且是 nn.Module, 跟随 .to(device)
        try:
            return next(self.projection.parameters()).device
        except StopIteration:
            return torch.device('cpu')

    def _encode_texts_raw(self, texts: List[str]) -> torch.Tensor:
        """
        编码文本为原始嵌入向量

        后端分派:
          - "st": 用 sentence-transformers
          - "fallback": 字符级 hash 编码 + 可学习嵌入

        Returns:
            [B, raw_dim] raw embeddings
        """
        device = self._get_device()

        if self._backend == "st":
            return self._encode_with_st(texts, device)
        else:
            return self._encode_with_fallback(texts, device)

    def _encode_with_st(self, texts: List[str], device: torch.device) -> torch.Tensor:
        """sentence-transformers 后端"""
        if self.training and not self.freeze_base:
            features = self._st_model.tokenize(texts)
            features = {k: v.to(device) for k, v in features.items()}
            embeddings = self._st_model(features)['sentence_embedding']
        else:
            with torch.no_grad():
                emb = self._st_model.encode(
                    texts,
                    convert_to_tensor=True,
                    show_progress_bar=False,
                )
                embeddings = emb.to(device) if isinstance(emb, torch.Tensor) else torch.tensor(emb, device=device)
        return embeddings

    def _encode_with_fallback(self, texts: List[str], device: torch.device) -> torch.Tensor:
        """
        fallback 后端: 字符级 hash → embedding → mean pool

        不依赖任何外部 tokenizer; 用 Python ord() 将字符映射到 vocab index
        精度不如 sentence-transformers, 但保证:
          1. 不同文本产出不同嵌入 (可区分)
          2. 可微, 支持训练
          3. 零外部依赖
        """
        B = len(texts)
        max_len = min(self.max_length, max(len(t) for t in texts) if texts else 1)

        # 构造 token ids: ord(char) % vocab_size
        token_ids = torch.zeros(B, max_len, dtype=torch.long, device=device)
        mask = torch.zeros(B, max_len, dtype=torch.float32, device=device)

        for i, text in enumerate(texts):
            for j, ch in enumerate(text[:max_len]):
                token_ids[i, j] = ord(ch) % self._FALLBACK_VOCAB_SIZE
                mask[i, j] = 1.0

            # 至少有一个 token (空字符串用 [0])
            if len(text) == 0:
                mask[i, 0] = 1.0

        # embed + pool
        embedded = self._fallback_embed(token_ids)  # [B, L, H]
        embedded = self._fallback_pool(embedded)     # [B, L, H]

        # masked mean pooling
        mask_expanded = mask.unsqueeze(-1)  # [B, L, 1]
        summed = (embedded * mask_expanded).sum(dim=1)  # [B, H]
        counts = mask_expanded.sum(dim=1).clamp(min=1)  # [B, 1]
        pooled = summed / counts  # [B, H]

        return pooled

    def forward(
        self,
        texts: List[str],                        # 文本列表
    ) -> torch.Tensor:
        """
        将文本编码为嵌入向量

        Returns:
            S_Y: [B, D] 目标嵌入, D=embed_dim
        """
        raw = self._encode_texts_raw(texts)       # [B, raw_dim]
        projected = self.projection(raw)          # [B, embed_dim]
        # L2 归一化 (与视觉嵌入对齐, 用于 InfoNCE)
        projected = F.normalize(projected, dim=-1)
        return projected

    def encode_class_labels(
        self,
        class_ids: torch.Tensor,                 # [B, num_objects] 类别 ID
        class_names: dict,                        # {0: "D00", 1: "D10", ...}
    ) -> torch.Tensor:
        """
        将 YOLO 格式的类别 ID 转换为文本嵌入
        构造结构化描述后编码

        策略: 将每个样本的所有目标类别名连接为一句话
          e.g. class_ids = [[0, 2]] → "Longitudinal Crack, Alligator Crack"

        Returns:
            S_Y: [B, D] 类别嵌入
        """
        batch_texts = []
        B = class_ids.shape[0]
        for b in range(B):
            ids = class_ids[b]
            # 过滤 padding (-1 或无效 ID)
            valid_ids = [int(i) for i in ids if int(i) in class_names]
            if valid_ids:
                names = [class_names[i] for i in valid_ids]
                # 去重保序
                seen = set()
                unique_names = []
                for n in names:
                    if n not in seen:
                        seen.add(n)
                        unique_names.append(n)
                text = ", ".join(unique_names)
            else:
                text = "no damage"
            batch_texts.append(text)

        return self.forward(batch_texts)

    def encode_spatial_description(
        self,
        bboxes: torch.Tensor,                    # [B, num_objects, 4] (cx, cy, w, h)
        class_ids: torch.Tensor,                  # [B, num_objects]
        class_names: dict,
        img_size: Tuple[int, int] = (640, 640),
    ) -> torch.Tensor:
        """
        将 bbox + 类别组合为空间描述文本, 然后编码
        例: "A longitudinal crack at the upper-left region, a pothole at the center"

        Returns:
            S_Y: [B, D] 空间结构化嵌入
        """
        batch_texts = []
        B = bboxes.shape[0]

        for b in range(B):
            descriptions = []
            num_objects = bboxes.shape[1]
            for j in range(num_objects):
                cls_id = int(class_ids[b, j])
                if cls_id < 0 or cls_id not in class_names:
                    continue  # padding

                cx, cy = float(bboxes[b, j, 0]), float(bboxes[b, j, 1])
                # 归一化坐标 → 3x3 区域网格
                row = min(int(cy * 3), 2)
                col = min(int(cx * 3), 2)
                region = self._REGION_MAP.get((row, col), "center")
                name = class_names[cls_id].lower()

                # 冠词
                article = "an" if name[0] in "aeiou" else "a"
                descriptions.append(f"{article} {name} at the {region}")

            if descriptions:
                text = ", ".join(descriptions)
            else:
                text = "no damage detected"
            batch_texts.append(text)

        return self.forward(batch_texts)

    def get_lr_params(self) -> List[dict]:
        """
        返回分组参数, 基础模型用 lr_multiplier 缩放学习率
        投影头用正常学习率
        """
        param_groups = []

        # 基础模型参数 (低学习率, 仅 st 后端)
        if self._backend == "st" and self._st_model is not None and not self.freeze_base:
            param_groups.append({
                'params': list(self._st_model.parameters()),
                'lr_multiplier': self.lr_multiplier,
                'name': 'text_encoder_base',
            })

        # fallback 后端的 embed + pool (正常学习率)
        if self._backend == "fallback":
            fallback_params = []
            if self._fallback_embed is not None:
                fallback_params.extend(list(self._fallback_embed.parameters()))
            if self._fallback_pool is not None:
                fallback_params.extend(list(self._fallback_pool.parameters()))
            if fallback_params:
                param_groups.append({
                    'params': fallback_params,
                    'lr_multiplier': 1.0,
                    'name': 'text_encoder_fallback',
                })

        # 投影头参数 (正常学习率)
        if self.projection is not None and not isinstance(self.projection, nn.Identity):
            param_groups.append({
                'params': list(self.projection.parameters()),
                'lr_multiplier': 1.0,
                'name': 'text_encoder_projection',
            })

        return param_groups
