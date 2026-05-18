#!/usr/bin/env python3
"""
道路裂缝检测 YOLO11 训练脚本
Road Damage Detection (RDD) Training Script

数据集包含 5 类道路损伤:
- D00: 纵向裂缝 (Longitudinal Crack)
- D10: 横向裂缝 (Transverse Crack)  
- D20: 网状裂缝 (Alligator Crack)
- D40: 坑洞 (Pothole)
- D44: 其他损伤 (Other Damage)
"""

import os
import argparse
from pathlib import Path
from datetime import datetime

# 设置项目根目录
PROJECT_ROOT = Path(__file__).parent.absolute()


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='YOLO11 道路裂缝检测训练脚本')
    
    # 基础配置
    parser.add_argument('--model', type=str, default='yolo11l.pt',
                        help='预训练模型路径 (default: yolo11l.pt)')
    parser.add_argument('--data', type=str, default='rdd_crack.yaml',
                        help='数据集配置文件 (default: rdd_crack.yaml)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数 (default: 100)')
    parser.add_argument('--batch', type=int, default=16,
                        help='批次大小, -1 表示自动调整 (default: 16)')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='输入图像尺寸 (default: 640)')
    
    # 训练超参数
    parser.add_argument('--lr0', type=float, default=0.01,
                        help='初始学习率 (default: 0.01)')
    parser.add_argument('--lrf', type=float, default=0.01,
                        help='最终学习率倍数 (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.937,
                        help='SGD 动量 (default: 0.937)')
    parser.add_argument('--weight_decay', type=float, default=0.0005,
                        help='权重衰减 (default: 0.0005)')
    parser.add_argument('--warmup_epochs', type=float, default=3.0,
                        help='预热轮数 (default: 3.0)')
    
    # 数据增强
    parser.add_argument('--hsv_h', type=float, default=0.015,
                        help='HSV 色调增强 (default: 0.015)')
    parser.add_argument('--hsv_s', type=float, default=0.7,
                        help='HSV 饱和度增强 (default: 0.7)')
    parser.add_argument('--hsv_v', type=float, default=0.4,
                        help='HSV 亮度增强 (default: 0.4)')
    parser.add_argument('--degrees', type=float, default=0.0,
                        help='旋转角度范围 (default: 0.0)')
    parser.add_argument('--translate', type=float, default=0.1,
                        help='平移范围 (default: 0.1)')
    parser.add_argument('--scale', type=float, default=0.5,
                        help='缩放范围 (default: 0.5)')
    parser.add_argument('--shear', type=float, default=0.0,
                        help='剪切角度 (default: 0.0)')
    parser.add_argument('--flipud', type=float, default=0.0,
                        help='上下翻转概率 (default: 0.0)')
    parser.add_argument('--fliplr', type=float, default=0.5,
                        help='左右翻转概率 (default: 0.5)')
    parser.add_argument('--mosaic', type=float, default=1.0,
                        help='Mosaic 增强概率 (default: 1.0)')
    parser.add_argument('--mixup', type=float, default=0.0,
                        help='Mixup 增强概率 (default: 0.0)')
    parser.add_argument('--copy_paste', type=float, default=0.0,
                        help='Copy-Paste 增强概率 (default: 0.0)')
    
    # 设备配置
    parser.add_argument('--device', type=str, default='',
                        help='训练设备 cuda:0, 0,1,2,3, cpu (default: 自动选择)')
    parser.add_argument('--workers', type=int, default=8,
                        help='数据加载工作进程数 (default: 8)')
    
    # 输出配置
    parser.add_argument('--project', type=str, default='runs/detect',
                        help='项目保存目录 (default: runs/detect)')
    parser.add_argument('--name', type=str, default='rdd_train',
                        help='实验名称 (default: rdd_train)')
    parser.add_argument('--exist_ok', action='store_true',
                        help='是否覆盖已有实验目录')
    
    # 训练控制
    parser.add_argument('--resume', type=str, default='',
                        help='断点续训检查点路径')
    parser.add_argument('--patience', type=int, default=50,
                        help='早停耐心值 (default: 50)')
    parser.add_argument('--save_period', type=int, default=10,
                        help='每隔多少轮保存检查点 (default: 10)')
    parser.add_argument('--cache', action='store_true',
                        help='是否缓存图像到内存')
    parser.add_argument('--amp', action='store_true', default=True,
                        help='使用自动混合精度训练')
    parser.add_argument('--cos_lr', action='store_true',
                        help='使用余弦学习率调度')
    parser.add_argument('--close_mosaic', type=int, default=10,
                        help='最后 N 轮关闭 Mosaic (default: 10)')
    
    # 验证和评估
    parser.add_argument('--val', action='store_true', default=True,
                        help='训练时进行验证')
    parser.add_argument('--plots', action='store_true', default=True,
                        help='保存训练曲线图')
    
    return parser.parse_args()


def update_data_config(data_path: str) -> str:
    """
    更新数据集配置文件中的 path 字段
    
    Args:
        data_path: 原始数据配置文件路径
        
    Returns:
        更新后的配置文件路径
    """
    import yaml
    
    # 读取原始配置
    with open(data_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 设置数据集根目录
    config['path'] = str(PROJECT_ROOT / 'RDD_SPLIT')
    
    # 保存到临时文件
    temp_config_path = PROJECT_ROOT / 'rdd_crack_train.yaml'
    with open(temp_config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
    
    print(f"✓ 数据集配置已更新: {temp_config_path}")
    print(f"  - 数据集路径: {config['path']}")
    print(f"  - 类别数量: {config['nc']}")
    print(f"  - 类别名称: {list(config['names'].values())}")
    
    return str(temp_config_path)


def main():
    """主训练函数"""
    args = parse_args()
    
    print("=" * 60)
    print("🚗 道路裂缝检测 YOLO11 训练")
    print("=" * 60)
    print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 检查依赖
    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ 请先安装 ultralytics: pip install ultralytics")
        return
    
    # 处理路径
    model_path = PROJECT_ROOT / args.model
    data_path = PROJECT_ROOT / args.data
    
    # 检查文件是否存在
    if not model_path.exists():
        print(f"❌ 预训练模型不存在: {model_path}")
        print("   请下载 YOLO11 预训练权重或指定正确路径")
        return
    
    if not data_path.exists():
        print(f"❌ 数据配置文件不存在: {data_path}")
        return
    
    # 更新数据配置
    data_config = update_data_config(str(data_path))
    
    # 加载模型
    print(f"\n📦 加载预训练模型: {model_path}")
    if args.resume:
        model = YOLO(args.resume)
        print(f"   断点续训自: {args.resume}")
    else:
        model = YOLO(str(model_path))
    
    # 打印训练配置
    print("\n📋 训练配置:")
    print(f"   - 训练轮数: {args.epochs}")
    print(f"   - 批次大小: {args.batch}")
    print(f"   - 图像尺寸: {args.imgsz}")
    print(f"   - 初始学习率: {args.lr0}")
    print(f"   - 设备: {args.device if args.device else '自动'}")
    print(f"   - 工作进程: {args.workers}")
    print(f"   - 混合精度: {args.amp}")
    print()
    
    # 开始训练
    print("🚀 开始训练...")
    print("-" * 60)
    
    results = model.train(
        # 基础配置
        data=data_config,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        
        # 学习率配置
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        
        # 数据增强
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        shear=args.shear,
        flipud=args.flipud,
        fliplr=args.fliplr,
        mosaic=args.mosaic,
        mixup=args.mixup,
        copy_paste=args.copy_paste,
        
        # 设备配置
        device=args.device if args.device else None,
        workers=args.workers,
        
        # 输出配置
        project=str(PROJECT_ROOT / args.project),
        name=args.name,
        exist_ok=args.exist_ok,
        
        # 训练控制
        patience=args.patience,
        save_period=args.save_period,
        cache=args.cache,
        amp=args.amp,
        cos_lr=args.cos_lr,
        close_mosaic=args.close_mosaic,
        
        # 验证
        val=args.val,
        plots=args.plots,
        
        # 其他
        verbose=True,
        seed=42,
    )
    
    print("-" * 60)
    print("\n✅ 训练完成!")
    print(f"⏰ 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 打印结果路径
    save_dir = Path(results.save_dir) if hasattr(results, 'save_dir') else None
    if save_dir and save_dir.exists():
        print(f"\n📁 结果保存位置: {save_dir}")
        print(f"   - 最佳模型: {save_dir / 'weights' / 'best.pt'}")
        print(f"   - 最终模型: {save_dir / 'weights' / 'last.pt'}")
    
    return results


if __name__ == '__main__':
    main()

