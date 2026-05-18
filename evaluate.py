#!/usr/bin/env python3
"""
道路裂缝检测 YOLO11 评估脚本
Road Damage Detection (RDD) Evaluation Script

评估指标:
- mAP50-95: 主要指标，IoU 从 0.5 到 0.95 的平均 AP
- mAP50: IoU=0.5 时的 mAP
- mAP75: IoU=0.75 时的 mAP
- Precision: 精确率
- Recall: 召回率
- F1-Score: F1 分数
- 每个类别的详细指标
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np

# 设置项目根目录
PROJECT_ROOT = Path(__file__).parent.absolute()

# 类别名称
CLASS_NAMES = {
    0: 'D00 (纵向裂缝)',
    1: 'D10 (横向裂缝)',
    2: 'D20 (网状裂缝)',
    3: 'D40 (坑洞)',
    4: 'D44 (其他损伤)'
}


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='YOLO11 道路裂缝检测评估脚本')
    
    parser.add_argument('--model', type=str, required=True,
                        help='模型权重路径 (如: runs/detect/rdd_train/weights/best.pt)')
    parser.add_argument('--data', type=str, default='rdd_crack.yaml',
                        help='数据集配置文件 (default: rdd_crack.yaml)')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='评估数据集分割 (default: test)')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='输入图像尺寸 (default: 640)')
    parser.add_argument('--batch', type=int, default=16,
                        help='批次大小 (default: 16)')
    parser.add_argument('--conf', type=float, default=0.001,
                        help='置信度阈值 (default: 0.001)')
    parser.add_argument('--iou', type=float, default=0.6,
                        help='NMS IoU 阈值 (default: 0.6)')
    parser.add_argument('--device', type=str, default='',
                        help='评估设备 (default: 自动选择)')
    parser.add_argument('--workers', type=int, default=8,
                        help='数据加载工作进程数 (default: 8)')
    parser.add_argument('--save_json', action='store_true',
                        help='保存 COCO 格式结果 JSON')
    parser.add_argument('--save_txt', action='store_true',
                        help='保存预测结果为 TXT')
    parser.add_argument('--plots', action='store_true', default=True,
                        help='生成评估图表 (混淆矩阵、PR曲线等)')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='详细输出')
    parser.add_argument('--output', type=str, default='',
                        help='结果保存目录 (default: 模型同级目录)')
    
    return parser.parse_args()


def update_data_config(data_path: str, split: str) -> str:
    """更新数据集配置文件"""
    import yaml
    
    with open(data_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    config['path'] = str(PROJECT_ROOT / 'RDD_SPLIT')
    
    temp_config_path = PROJECT_ROOT / f'rdd_crack_eval_{split}.yaml'
    with open(temp_config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
    
    return str(temp_config_path)


def calculate_f1(precision: float, recall: float) -> float:
    """计算 F1 分数"""
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def print_metrics_table(results, class_names: dict):
    """打印详细指标表格"""
    print("\n" + "=" * 80)
    print("📊 详细评估结果")
    print("=" * 80)
    
    # 获取指标
    metrics = results.results_dict
    
    # 总体指标
    print("\n🎯 总体指标:")
    print("-" * 50)
    
    # 核心指标
    map50_95 = metrics.get('metrics/mAP50-95(B)', 0)
    map50 = metrics.get('metrics/mAP50(B)', 0)
    precision = metrics.get('metrics/precision(B)', 0)
    recall = metrics.get('metrics/recall(B)', 0)
    f1 = calculate_f1(precision, recall)
    
    print(f"  {'指标':<25} {'值':>10}")
    print(f"  {'-' * 35}")
    print(f"  {'mAP50-95 (主要指标)':<25} {map50_95:>10.4f}")
    print(f"  {'mAP50':<25} {map50:>10.4f}")
    print(f"  {'Precision (精确率)':<25} {precision:>10.4f}")
    print(f"  {'Recall (召回率)':<25} {recall:>10.4f}")
    print(f"  {'F1-Score':<25} {f1:>10.4f}")
    
    # 尝试获取每类指标
    if hasattr(results, 'box'):
        box = results.box
        if hasattr(box, 'ap50') and hasattr(box, 'ap'):
            print("\n📋 每类别指标:")
            print("-" * 80)
            print(f"  {'类别':<20} {'AP50':>10} {'AP50-95':>12} {'Precision':>12} {'Recall':>10} {'F1':>10}")
            print(f"  {'-' * 74}")
            
            ap50_per_class = box.ap50
            ap_per_class = box.ap
            
            # 如果有每类的 P 和 R
            p_per_class = box.p if hasattr(box, 'p') else [0] * len(class_names)
            r_per_class = box.r if hasattr(box, 'r') else [0] * len(class_names)
            
            for i, (cls_id, cls_name) in enumerate(class_names.items()):
                if i < len(ap50_per_class):
                    ap50_cls = ap50_per_class[i]
                    ap_cls = ap_per_class[i]
                    p_cls = p_per_class[i] if i < len(p_per_class) else 0
                    r_cls = r_per_class[i] if i < len(r_per_class) else 0
                    f1_cls = calculate_f1(p_cls, r_cls)
                    
                    print(f"  {cls_name:<20} {ap50_cls:>10.4f} {ap_cls:>12.4f} {p_cls:>12.4f} {r_cls:>10.4f} {f1_cls:>10.4f}")
    
    print()
    return {
        'mAP50-95': map50_95,
        'mAP50': map50,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def save_results(results, output_dir: Path, args):
    """保存评估结果"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics = results.results_dict
    
    # 获取核心指标
    map50_95 = metrics.get('metrics/mAP50-95(B)', 0)
    map50 = metrics.get('metrics/mAP50(B)', 0)
    precision = metrics.get('metrics/precision(B)', 0)
    recall = metrics.get('metrics/recall(B)', 0)
    f1 = calculate_f1(precision, recall)
    
    # 构建结果字典
    results_dict = {
        'evaluation_info': {
            'model': str(args.model),
            'dataset': str(args.data),
            'split': args.split,
            'image_size': args.imgsz,
            'conf_threshold': args.conf,
            'iou_threshold': args.iou,
            'timestamp': datetime.now().isoformat()
        },
        'overall_metrics': {
            'mAP50-95': float(map50_95),
            'mAP50': float(map50),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1)
        },
        'per_class_metrics': {}
    }
    
    # 添加每类指标
    if hasattr(results, 'box'):
        box = results.box
        if hasattr(box, 'ap50') and hasattr(box, 'ap'):
            ap50_per_class = box.ap50
            ap_per_class = box.ap
            p_per_class = box.p if hasattr(box, 'p') else [0] * len(CLASS_NAMES)
            r_per_class = box.r if hasattr(box, 'r') else [0] * len(CLASS_NAMES)
            
            for i, (cls_id, cls_name) in enumerate(CLASS_NAMES.items()):
                if i < len(ap50_per_class):
                    results_dict['per_class_metrics'][cls_name] = {
                        'AP50': float(ap50_per_class[i]),
                        'AP50-95': float(ap_per_class[i]),
                        'precision': float(p_per_class[i]) if i < len(p_per_class) else 0,
                        'recall': float(r_per_class[i]) if i < len(r_per_class) else 0,
                        'f1_score': float(calculate_f1(
                            p_per_class[i] if i < len(p_per_class) else 0,
                            r_per_class[i] if i < len(r_per_class) else 0
                        ))
                    }
    
    # 保存 JSON
    json_path = output_dir / 'evaluation_results.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)
    
    print(f"📁 评估结果已保存: {json_path}")
    
    # 保存文本报告
    txt_path = output_dir / 'evaluation_report.txt'
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("道路裂缝检测模型评估报告\n")
        f.write("Road Damage Detection Evaluation Report\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"模型: {args.model}\n")
        f.write(f"数据集: {args.split}\n")
        f.write(f"图像尺寸: {args.imgsz}\n\n")
        
        f.write("-" * 40 + "\n")
        f.write("总体指标\n")
        f.write("-" * 40 + "\n")
        f.write(f"mAP50-95 (主要指标): {map50_95:.4f}\n")
        f.write(f"mAP50:               {map50:.4f}\n")
        f.write(f"Precision:           {precision:.4f}\n")
        f.write(f"Recall:              {recall:.4f}\n")
        f.write(f"F1-Score:            {f1:.4f}\n\n")
        
        if results_dict['per_class_metrics']:
            f.write("-" * 40 + "\n")
            f.write("每类别指标\n")
            f.write("-" * 40 + "\n")
            for cls_name, cls_metrics in results_dict['per_class_metrics'].items():
                f.write(f"\n{cls_name}:\n")
                f.write(f"  AP50:      {cls_metrics['AP50']:.4f}\n")
                f.write(f"  AP50-95:   {cls_metrics['AP50-95']:.4f}\n")
                f.write(f"  Precision: {cls_metrics['precision']:.4f}\n")
                f.write(f"  Recall:    {cls_metrics['recall']:.4f}\n")
                f.write(f"  F1-Score:  {cls_metrics['f1_score']:.4f}\n")
    
    print(f"📄 评估报告已保存: {txt_path}")
    
    return results_dict


def main():
    """主评估函数"""
    args = parse_args()
    
    print("=" * 60)
    print("🔍 道路裂缝检测 YOLO11 模型评估")
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
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / args.model
    
    data_path = PROJECT_ROOT / args.data
    
    # 检查文件
    if not model_path.exists():
        print(f"❌ 模型文件不存在: {model_path}")
        return
    
    if not data_path.exists():
        print(f"❌ 数据配置文件不存在: {data_path}")
        return
    
    # 确定输出目录
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = model_path.parent.parent / f'evaluate_{args.split}'
    
    # 更新数据配置
    data_config = update_data_config(str(data_path), args.split)
    
    # 加载模型
    print(f"📦 加载模型: {model_path}")
    model = YOLO(str(model_path))
    
    # 打印评估配置
    print(f"\n📋 评估配置:")
    print(f"   - 数据集分割: {args.split}")
    print(f"   - 图像尺寸: {args.imgsz}")
    print(f"   - 批次大小: {args.batch}")
    print(f"   - 置信度阈值: {args.conf}")
    print(f"   - IoU 阈值: {args.iou}")
    print(f"   - 设备: {args.device if args.device else '自动'}")
    print()
    
    # 开始评估
    print("🚀 开始评估...")
    print("-" * 60)
    
    results = model.val(
        data=data_config,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        device=args.device if args.device else None,
        workers=args.workers,
        save_json=args.save_json,
        save_txt=args.save_txt,
        plots=args.plots,
        verbose=args.verbose,
    )
    
    # 打印详细指标
    metrics_summary = print_metrics_table(results, CLASS_NAMES)
    
    # 保存结果
    print("-" * 60)
    results_dict = save_results(results, output_dir, args)
    
    print()
    print("=" * 60)
    print("✅ 评估完成!")
    print(f"⏰ 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # 打印核心指标总结
    print("\n📈 核心指标总结:")
    print(f"   mAP50-95: {metrics_summary['mAP50-95']:.4f}")
    print(f"   mAP50:    {metrics_summary['mAP50']:.4f}")
    print(f"   F1-Score: {metrics_summary['f1']:.4f}")
    print()
    
    return results


if __name__ == '__main__':
    main()

