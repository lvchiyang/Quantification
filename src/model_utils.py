"""
模型工具函数
包含模型验证、保存加载、分析等实用功能
"""

import torch
import torch.nn as nn
import numpy as np
import json
import os
from typing import Dict, Any, Optional, Tuple, List
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns

from .config import ModelArgs
from .transformer import Transformer
from .utils import count_parameters


def validate_model_architecture(model: Transformer, args: ModelArgs) -> Dict[str, Any]:
    """
    验证模型架构的正确性
    
    Args:
        model: Transformer 模型
        args: 模型配置
        
    Returns:
        验证结果字典
    """
    results = {
        "passed": True,
        "errors": [],
        "warnings": [],
        "info": {}
    }
    
    try:
        # 1. 检查模型参数数量
        total_params = count_parameters(model)
        results["info"]["total_parameters"] = total_params
        
        # 2. 检查输入输出维度
        batch_size, seq_len = 2, 128
        dummy_input = torch.randint(0, args.vocab_size, (batch_size, seq_len))
        
        with torch.no_grad():
            outputs = model(dummy_input)
            logits = outputs["logits"]
            
            # 检查输出形状
            expected_shape = (batch_size, seq_len, args.vocab_size)
            if logits.shape != expected_shape:
                results["errors"].append(
                    f"Output shape mismatch: got {logits.shape}, expected {expected_shape}"
                )
                results["passed"] = False
            
            # 检查损失计算
            loss_outputs = model(dummy_input, labels=dummy_input)
            if loss_outputs["loss"] is None:
                results["errors"].append("Loss computation failed")
                results["passed"] = False
            else:
                results["info"]["sample_loss"] = loss_outputs["loss"].item()
        
        # 3. 检查梯度流
        model.train()
        dummy_input = torch.randint(0, args.vocab_size, (1, 32))
        outputs = model(dummy_input, labels=dummy_input)
        loss = outputs["loss"]
        loss.backward()
        
        # 检查是否有梯度
        has_gradients = False
        zero_gradients = 0
        total_params_with_grad = 0
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                total_params_with_grad += 1
                if param.grad is not None:
                    has_gradients = True
                    if param.grad.abs().sum() == 0:
                        zero_gradients += 1
        
        if not has_gradients:
            results["errors"].append("No gradients computed")
            results["passed"] = False
        
        if zero_gradients > total_params_with_grad * 0.5:
            results["warnings"].append(
                f"Many parameters have zero gradients: {zero_gradients}/{total_params_with_grad}"
            )
        
        results["info"]["parameters_with_gradients"] = total_params_with_grad
        results["info"]["zero_gradient_parameters"] = zero_gradients
        
        # 4. 检查注意力机制
        # 这里可以添加更多特定的检查
        
        # 5. 内存使用检查
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            memory_before = torch.cuda.memory_allocated()
            
            # 前向传播
            with torch.no_grad():
                model(dummy_input)
            
            memory_after = torch.cuda.memory_allocated()
            results["info"]["gpu_memory_usage_mb"] = (memory_after - memory_before) / 1024**2
        
    except Exception as e:
        results["errors"].append(f"Validation failed with exception: {str(e)}")
        results["passed"] = False
    
    return results


def analyze_model_weights(model: Transformer) -> Dict[str, Any]:
    """
    分析模型权重分布
    
    Args:
        model: Transformer 模型
        
    Returns:
        权重分析结果
    """
    analysis = {
        "layer_stats": {},
        "overall_stats": {},
        "potential_issues": []
    }
    
    all_weights = []
    
    for name, param in model.named_parameters():
        if param.requires_grad and param.dim() >= 2:  # 只分析权重矩阵
            weights = param.data.cpu().numpy().flatten()
            all_weights.extend(weights)
            
            # 计算统计信息
            stats = {
                "mean": float(np.mean(weights)),
                "std": float(np.std(weights)),
                "min": float(np.min(weights)),
                "max": float(np.max(weights)),
                "shape": list(param.shape),
                "num_params": param.numel()
            }
            
            # 检查潜在问题
            if abs(stats["mean"]) > 0.1:
                analysis["potential_issues"].append(f"{name}: Large mean ({stats['mean']:.4f})")
            
            if stats["std"] > 1.0:
                analysis["potential_issues"].append(f"{name}: Large std ({stats['std']:.4f})")
            
            if stats["std"] < 0.01:
                analysis["potential_issues"].append(f"{name}: Small std ({stats['std']:.4f})")
            
            analysis["layer_stats"][name] = stats
    
    # 整体统计
    all_weights = np.array(all_weights)
    analysis["overall_stats"] = {
        "mean": float(np.mean(all_weights)),
        "std": float(np.std(all_weights)),
        "min": float(np.min(all_weights)),
        "max": float(np.max(all_weights)),
        "total_parameters": len(all_weights)
    }
    
    return analysis


def plot_weight_distributions(model: Transformer, save_path: Optional[str] = None):
    """
    绘制权重分布图
    
    Args:
        model: Transformer 模型
        save_path: 保存路径
    """
    # 收集权重
    layer_weights = {}
    for name, param in model.named_parameters():
        if param.requires_grad and param.dim() >= 2:
            layer_weights[name] = param.data.cpu().numpy().flatten()
    
    # 创建子图
    n_layers = len(layer_weights)
    n_cols = min(4, n_layers)
    n_rows = (n_layers + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
    if n_layers == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # 绘制每层的权重分布
    for i, (name, weights) in enumerate(layer_weights.items()):
        if i < len(axes):
            axes[i].hist(weights, bins=50, alpha=0.7, density=True)
            axes[i].set_title(name.split('.')[-2] + '.' + name.split('.')[-1])
            axes[i].set_xlabel('Weight Value')
            axes[i].set_ylabel('Density')
            axes[i].grid(True, alpha=0.3)
    
    # 隐藏多余的子图
    for i in range(n_layers, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Weight distribution plot saved to {save_path}")
    
    plt.show()


def benchmark_model_speed(
    model: Transformer, 
    batch_sizes: List[int] = [1, 4, 8, 16],
    seq_lengths: List[int] = [128, 256, 512, 1024],
    num_runs: int = 10
) -> Dict[str, Any]:
    """
    基准测试模型速度
    
    Args:
        model: Transformer 模型
        batch_sizes: 批次大小列表
        seq_lengths: 序列长度列表
        num_runs: 运行次数
        
    Returns:
        基准测试结果
    """
    import time
    
    device = next(model.parameters()).device
    model.eval()
    
    results = {
        "device": str(device),
        "forward_pass_times": {},
        "generation_times": {},
        "throughput": {}
    }
    
    vocab_size = model.args.vocab_size
    
    print("Running speed benchmarks...")
    
    # 前向传播基准测试
    for batch_size in batch_sizes:
        for seq_len in seq_lengths:
            if seq_len > model.args.max_seq_len:
                continue
                
            key = f"bs{batch_size}_seq{seq_len}"
            
            # 创建随机输入
            dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
            
            # 预热
            with torch.no_grad():
                for _ in range(3):
                    model(dummy_input)
            
            # 同步 GPU
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            # 测量时间
            times = []
            for _ in range(num_runs):
                start_time = time.time()
                
                with torch.no_grad():
                    outputs = model(dummy_input)
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
                end_time = time.time()
                times.append(end_time - start_time)
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            
            results["forward_pass_times"][key] = {
                "avg_time_ms": avg_time * 1000,
                "std_time_ms": std_time * 1000,
                "tokens_per_second": (batch_size * seq_len) / avg_time
            }
            
            print(f"Forward pass {key}: {avg_time*1000:.2f}±{std_time*1000:.2f}ms, "
                  f"{(batch_size * seq_len) / avg_time:.0f} tokens/s")
    
    # 生成基准测试
    print("\nTesting generation speed...")
    for batch_size in [1, 2, 4]:
        if batch_size > max(batch_sizes):
            continue
            
        key = f"bs{batch_size}"
        
        # 创建输入
        prompt_len = 32
        dummy_input = torch.randint(0, vocab_size, (batch_size, prompt_len)).to(device)
        
        # 测量生成时间
        times = []
        for _ in range(min(num_runs, 5)):  # 生成测试运行次数较少
            start_time = time.time()
            
            with torch.no_grad():
                generated = model.generate(
                    dummy_input,
                    max_new_tokens=50,
                    do_sample=False  # 贪心解码更稳定
                )
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        results["generation_times"][key] = {
            "avg_time_ms": avg_time * 1000,
            "tokens_per_second": (batch_size * 50) / avg_time
        }
        
        print(f"Generation {key}: {avg_time*1000:.2f}ms, "
              f"{(batch_size * 50) / avg_time:.0f} tokens/s")
    
    return results


def save_model_info(model: Transformer, args: ModelArgs, save_dir: str):
    """
    保存模型信息到文件
    
    Args:
        model: Transformer 模型
        args: 模型配置
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 保存配置
    config_dict = {k: v for k, v in args.__dict__.items() if not k.startswith('_')}
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    # 2. 验证模型
    validation_results = validate_model_architecture(model, args)
    with open(os.path.join(save_dir, 'validation.json'), 'w') as f:
        json.dump(validation_results, f, indent=2)
    
    # 3. 分析权重
    weight_analysis = analyze_model_weights(model)
    with open(os.path.join(save_dir, 'weight_analysis.json'), 'w') as f:
        json.dump(weight_analysis, f, indent=2)
    
    # 4. 绘制权重分布
    plot_weight_distributions(model, os.path.join(save_dir, 'weight_distributions.png'))
    
    # 5. 基准测试（如果有 GPU）
    if torch.cuda.is_available():
        benchmark_results = benchmark_model_speed(model)
        with open(os.path.join(save_dir, 'benchmark.json'), 'w') as f:
            json.dump(benchmark_results, f, indent=2)
    
    print(f"Model information saved to {save_dir}")


def load_model_for_inference(checkpoint_path: str, device: str = "auto") -> Tuple[Transformer, ModelArgs, AutoTokenizer]:
    """
    加载模型用于推理
    
    Args:
        checkpoint_path: 检查点路径
        device: 设备
        
    Returns:
        (model, args, tokenizer)
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args = checkpoint['args']
    
    # 创建模型
    model = Transformer(args)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, args, tokenizer
