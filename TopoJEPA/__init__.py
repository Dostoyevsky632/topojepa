# TopoJEPA: Topology-Regularized Joint Embedding Predictive Architecture
# for Structure-Aware Representation Learning
#
# 核心思想:
#   1. 借鉴 JEPA 的嵌入空间预测范式 (不在像素空间重建, 而在嵌入空间预测)
#   2. 引入持久同调 (Persistent Homology) 作为嵌入空间的拓扑正则化
#   3. 理论上证明拓扑保真性是比均匀性更强的反坍塌条件
#   4. 以道路损伤检测为主验证场景, 同时验证通用性
#
# 项目结构:
#   models/     - 模型架构 (Encoder, Predictor, TopoJEPA)
#   losses/     - 损失函数 (JEPA Loss, Topo Loss, Detection Loss)
#   data/       - 数据加载 (RDD数据集适配, 多数据集接口)
#   utils/      - 工具函数 (EMA, 调度器, 可视化)
#   configs/    - 配置文件

__version__ = "0.1.0"
