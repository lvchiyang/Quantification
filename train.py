#!/usr/bin/env python3
"""
金融量化交易策略训练脚本 - 两阶段训练入口
第一阶段：训练价格预测网络
第二阶段：训练策略网络
"""

import subprocess
import sys
import os


def run_two_stage_training():
    """运行两阶段训练"""
    print("🚀 开始两阶段金融量化训练...")
    print("=" * 60)

    # 第一阶段：价格预测网络训练
    print("📈 第一阶段：训练价格预测网络")
    print("=" * 60)

    try:
        result = subprocess.run([
            sys.executable, "train_price_network.py"
        ], check=True, capture_output=False)
        print("✅ 价格预测网络训练完成!")
    except subprocess.CalledProcessError as e:
        print(f"❌ 价格预测网络训练失败: {e}")
        return False

    print("\n" + "=" * 60)

    # 第二阶段：策略网络训练
    print("🧠 第二阶段：训练策略网络")
    print("=" * 60)

    try:
        result = subprocess.run([
            sys.executable, "train_strategy_network.py"
        ], check=True, capture_output=False)
        print("✅ 策略网络训练完成!")
    except subprocess.CalledProcessError as e:
        print(f"❌ 策略网络训练失败: {e}")
        return False

    print("\n" + "=" * 60)
    print("🎉 两阶段训练全部完成!")
    print("📁 生成的模型文件:")
    print("  - best_price_network.pth    (价格预测网络)")
    print("  - best_strategy_network.pth (策略网络)")
    print("=" * 60)

    return True


def main():
    """主函数"""
    print("🎯 金融量化交易策略训练系统")
    print("💡 采用两阶段解耦训练方法:")
    print("   1️⃣ 价格预测网络 - 专门优化价格预测能力")
    print("   2️⃣ 策略网络 - 基于价格特征学习交易策略")
    print()

    # 检查是否要运行特定阶段
    if len(sys.argv) > 1:
        stage = sys.argv[1].lower()
        if stage == "price":
            print("🎯 只运行第一阶段：价格预测网络训练")
            subprocess.run([sys.executable, "train_price_network.py"])
        elif stage == "strategy":
            print("🎯 只运行第二阶段：策略网络训练")
            subprocess.run([sys.executable, "train_strategy_network.py"])
        else:
            print(f"❌ 未知参数: {stage}")
            print("💡 用法:")
            print("  python train.py        # 运行完整两阶段训练")
            print("  python train.py price  # 只训练价格网络")
            print("  python train.py strategy # 只训练策略网络")
    else:
        # 运行完整的两阶段训练
        success = run_two_stage_training()
        if not success:
            sys.exit(1)


if __name__ == "__main__":
    main()
