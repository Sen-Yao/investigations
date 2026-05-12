#!/usr/bin/env python3
"""
统一运行脚本 - 执行所有信息论分析

功能：
1. 信息熵分析
2. 互信息分析
3. Fisher 判别分析
4. 维度重要性分析

使用方式：
    python run_all_analysis.py --dataset photo
    python run_all_analysis.py --dataset photo --plot  # 包含图表
    python run_all_analysis.py --dataset all            # 所有数据集

作者：Nexus
日期：2026-03-31
"""

import argparse
import os
import sys
import subprocess
from datetime import datetime

# 脚本目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "outputs")
PLOT_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "plots")

# 分析脚本
SCRIPTS = [
    ("信息熵分析", "offset_entropy_analysis.py"),
    ("互信息分析", "offset_mutual_information.py"),
    ("Fisher 判别分析", "offset_fisher_analysis.py"),
    ("维度重要性分析", "offset_dimension_importance.py"),
]

DATASETS = ["photo", "elliptic", "tolokers"]


def run_script(script_path: str, args: list) -> bool:
    """
    运行单个脚本
    
    Args:
        script_path: 脚本路径
        args: 命令行参数
    
    Returns:
        success: 是否成功
    """
    cmd = ["python3", script_path] + args
    print(f"\n执行: {' '.join(cmd)}")
    print("-" * 60)
    
    result = subprocess.run(cmd, capture_output=False)
    
    return result.returncode == 0


def run_all_analyses(dataset: str, pp_k: int = 6, alpha: float = 0.1, 
                     plot: bool = False):
    """
    运行所有分析
    
    Args:
        dataset: 数据集名称
        pp_k: Hop 数量
        alpha: PPR 参数
        plot: 是否生成图表
    """
    print("=" * 60)
    print(f"Offset 信息论分析 - {dataset.upper()}")
    print("=" * 60)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    # 基础参数
    base_args = ["--dataset", dataset, "--pp_k", str(pp_k), "--alpha", str(alpha)]
    
    if plot:
        base_args.append("--plot")
    
    # 运行每个脚本
    results = {}
    for name, script in SCRIPTS:
        script_path = os.path.join(SCRIPT_DIR, script)
        
        if not os.path.exists(script_path):
            print(f"\n⚠️ 脚本不存在: {script_path}")
            results[name] = False
            continue
        
        success = run_script(script_path, base_args)
        results[name] = success
        
        if success:
            print(f"\n✅ {name} 完成")
        else:
            print(f"\n❌ {name} 失败")
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("执行结果汇总")
    print("=" * 60)
    for name, success in results.items():
        status = "✅ 成功" if success else "❌ 失败"
        print(f"{name}: {status}")
    
    success_count = sum(results.values())
    total_count = len(results)
    print(f"\n总计: {success_count}/{total_count} 成功")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="运行所有信息论分析")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["photo", "elliptic", "tolokers", "all"],
                        help="数据集名称 (all = 所有数据集)")
    parser.add_argument("--pp_k", type=int, default=6, help="Hop 数量")
    parser.add_argument("--alpha", type=float, default=0.1, help="PPR 参数")
    parser.add_argument("--plot", action="store_true", help="是否生成图表")
    
    args = parser.parse_args()
    
    datasets = DATASETS if args.dataset == "all" else [args.dataset]
    
    all_results = {}
    for dataset in datasets:
        print(f"\n{'=' * 60}")
        print(f"数据集: {dataset.upper()}")
        print("=" * 60)
        
        results = run_all_analyses(
            dataset=dataset,
            pp_k=args.pp_k,
            alpha=args.alpha,
            plot=args.plot
        )
        
        all_results[dataset] = results
    
    # 最终汇总
    print("\n" + "=" * 60)
    print("全部执行完成")
    print("=" * 60)
    
    for dataset, results in all_results.items():
        success_count = sum(results.values())
        total_count = len(results)
        print(f"{dataset}: {success_count}/{total_count} 成功")
    
    print("\n输出文件位置:")
    print(f"  JSON: {OUTPUT_DIR}")
    if args.plot:
        print(f"  图表: {PLOT_DIR}")


if __name__ == "__main__":
    main()