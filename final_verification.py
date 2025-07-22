#!/usr/bin/env python3
"""
最终验证脚本
验证所有实现的功能是否正常工作
"""

import sys
import os
import torch
import time
from typing import Dict, Any

# 添加 src 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def print_section(title: str):
    """打印章节标题"""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)

def print_subsection(title: str):
    """打印子章节标题"""
    print(f"\n--- {title} ---")

def test_imports() -> bool:
    """测试所有模块导入"""
    print_subsection("测试模块导入")
    
    try:
        # 核心模块
        from config import ModelArgs, ModelConfigs
        from utils import RMSNorm, precompute_freqs_cis, apply_rotary_emb
        from attention import MLA, MultiHeadAttention
        from feedforward import SwiGLU, TransformerBlock
        from transformer import Transformer
        from trainer import Trainer
        from model_utils import validate_model_architecture
        
        print("? 所有核心模块导入成功")
        return True
    except Exception as e:
        print(f"? 模块导入失败: {e}")
        return False

def test_model_configurations() -> bool:
    """测试不同模型配置"""
    print_subsection("测试模型配置")
    
    try:
        from config import ModelConfigs
        from transformer import Transformer
        
        configs = {
            "tiny": ModelConfigs.tiny(),
            "small": ModelConfigs.small(),
            "base": ModelConfigs.base()
        }
        
        for name, args in configs.items():
            model = Transformer(args)
            param_count = sum(p.numel() for p in model.parameters())
            print(f"? {name.upper()} 模型: {param_count:,} 参数")
        
        return True
    except Exception as e:
        print(f"? 模型配置测试失败: {e}")
        return False

def test_forward_pass() -> bool:
    """测试前向传播"""
    print_subsection("测试前向传播")
    
    try:
        from config import ModelConfigs
        from transformer import Transformer
        
        args = ModelConfigs.tiny()
        model = Transformer(args)
        
        # 测试不同批次大小和序列长度
        test_cases = [
            (1, 16),   # 单样本，短序列
            (2, 32),   # 小批次，中等序列
            (4, 64),   # 中等批次，长序列
        ]
        
        for batch_size, seq_len in test_cases:
            input_ids = torch.randint(0, args.vocab_size, (batch_size, seq_len))
            
            with torch.no_grad():
                outputs = model(input_ids)
                logits = outputs["logits"]
            
            expected_shape = (batch_size, seq_len, args.vocab_size)
            assert logits.shape == expected_shape, f"形状不匹配: {logits.shape} vs {expected_shape}"
            assert torch.isfinite(logits).all(), "输出包含非有限值"
            
            print(f"? 批次大小 {batch_size}, 序列长度 {seq_len}: {logits.shape}")
        
        return True
    except Exception as e:
        print(f"? 前向传播测试失败: {e}")
        return False

def test_loss_computation() -> bool:
    """测试损失计算"""
    print_subsection("测试损失计算")
    
    try:
        from config import ModelConfigs
        from transformer import Transformer
        
        args = ModelConfigs.tiny()
        model = Transformer(args)
        
        input_ids = torch.randint(0, args.vocab_size, (2, 32))
        
        # 测试损失计算
        outputs = model(input_ids, labels=input_ids)
        loss = outputs["loss"]
        
        assert loss is not None, "损失为 None"
        assert torch.isfinite(loss), "损失不是有限值"
        assert loss.item() > 0, "损失应该为正数"
        
        print(f"? 损失计算成功: {loss.item():.4f}")
        
        # 测试梯度计算
        loss.backward()
        
        grad_count = 0
        for param in model.parameters():
            if param.requires_grad and param.grad is not None:
                grad_count += 1
        
        print(f"? 梯度计算成功: {grad_count} 个参数有梯度")
        
        return True
    except Exception as e:
        print(f"? 损失计算测试失败: {e}")
        return False

def test_text_generation() -> bool:
    """测试文本生成"""
    print_subsection("测试文本生成")
    
    try:
        from config import ModelConfigs
        from transformer import Transformer
        
        args = ModelConfigs.tiny()
        model = Transformer(args)
        model.eval()
        
        # 测试不同生成参数
        test_cases = [
            {"do_sample": False, "max_new_tokens": 10},  # 贪心解码
            {"do_sample": True, "temperature": 1.0, "max_new_tokens": 10},  # 采样
            {"do_sample": True, "top_k": 10, "max_new_tokens": 10},  # Top-K
            {"do_sample": True, "top_p": 0.9, "max_new_tokens": 10},  # Top-P
        ]
        
        for i, gen_kwargs in enumerate(test_cases):
            input_ids = torch.randint(0, args.vocab_size, (1, 5))
            
            with torch.no_grad():
                generated = model.generate(input_ids=input_ids, **gen_kwargs)
            
            assert generated.shape[0] == 1, "批次大小不正确"
            assert generated.shape[1] > input_ids.shape[1], "没有生成新的 token"
            assert (generated >= 0).all(), "生成的 token ID 为负数"
            assert (generated < args.vocab_size).all(), "生成的 token ID 超出词汇表"
            
            print(f"? 生成测试 {i+1}: {input_ids.shape[1]} -> {generated.shape[1]} tokens")
        
        return True
    except Exception as e:
        print(f"? 文本生成测试失败: {e}")
        return False

def test_model_validation() -> bool:
    """测试模型验证"""
    print_subsection("测试模型验证")
    
    try:
        from config import ModelConfigs
        from transformer import Transformer
        from model_utils import validate_model_architecture
        
        args = ModelConfigs.tiny()
        model = Transformer(args)
        
        validation_results = validate_model_architecture(model, args)
        
        if validation_results["passed"]:
            print("? 模型架构验证通过")
            print(f"  总参数数: {validation_results['info']['total_parameters']:,}")
            print(f"  样本损失: {validation_results['info']['sample_loss']:.4f}")
        else:
            print("? 模型架构验证失败")
            for error in validation_results["errors"]:
                print(f"  错误: {error}")
            return False
        
        return True
    except Exception as e:
        print(f"? 模型验证测试失败: {e}")
        return False

def test_component_functionality() -> bool:
    """测试各个组件功能"""
    print_subsection("测试组件功能")
    
    try:
        from config import ModelConfigs
        from utils import RMSNorm, precompute_freqs_cis
        from attention import MLA
        from feedforward import SwiGLU
        
        args = ModelConfigs.tiny()
        batch_size, seq_len = 2, 32
        
        # 测试 RMSNorm
        norm = RMSNorm(args.d_model)
        x = torch.randn(batch_size, seq_len, args.d_model)
        normed = norm(x)
        assert normed.shape == x.shape, "RMSNorm 输出形状不正确"
        print("? RMSNorm 测试通过")
        
        # 测试 RoPE
        freqs_cis = precompute_freqs_cis(args.qk_rope_head_dim, seq_len)
        assert freqs_cis.shape == (seq_len, args.qk_rope_head_dim // 2), "RoPE 频率形状不正确"
        print("? RoPE 测试通过")
        
        # 测试 MLA
        mla = MLA(args)
        attn_out = mla(x, freqs_cis)
        assert attn_out.shape == x.shape, "MLA 输出形状不正确"
        print("? MLA 测试通过")
        
        # 测试 SwiGLU
        ffn = SwiGLU(args)
        ffn_out = ffn(x)
        assert ffn_out.shape == x.shape, "SwiGLU 输出形状不正确"
        print("? SwiGLU 测试通过")
        
        return True
    except Exception as e:
        print(f"? 组件功能测试失败: {e}")
        return False

def test_performance() -> bool:
    """测试性能"""
    print_subsection("测试性能")
    
    try:
        from config import ModelConfigs
        from transformer import Transformer
        
        args = ModelConfigs.tiny()
        model = Transformer(args)
        model.eval()
        
        # 测试前向传播速度
        input_ids = torch.randint(0, args.vocab_size, (4, 128))
        
        # 预热
        with torch.no_grad():
            for _ in range(3):
                model(input_ids)
        
        # 测量时间
        start_time = time.time()
        with torch.no_grad():
            for _ in range(10):
                outputs = model(input_ids)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 10
        tokens_per_second = (4 * 128) / avg_time
        
        print(f"? 前向传播性能: {avg_time*1000:.2f}ms, {tokens_per_second:.0f} tokens/s")
        
        # 测试生成速度
        input_ids = torch.randint(0, args.vocab_size, (1, 10))
        
        start_time = time.time()
        with torch.no_grad():
            generated = model.generate(input_ids, max_new_tokens=20, do_sample=False)
        end_time = time.time()
        
        gen_time = end_time - start_time
        gen_tokens_per_second = 20 / gen_time
        
        print(f"? 生成性能: {gen_time*1000:.2f}ms, {gen_tokens_per_second:.1f} tokens/s")
        
        return True
    except Exception as e:
        print(f"? 性能测试失败: {e}")
        return False

def main():
    """主函数"""
    print_section("Decoder-only Transformer with MLA - 最终验证")
    
    print("这个验证脚本将测试所有实现的功能...")
    
    # 测试列表
    tests = [
        ("模块导入", test_imports),
        ("模型配置", test_model_configurations),
        ("前向传播", test_forward_pass),
        ("损失计算", test_loss_computation),
        ("文本生成", test_text_generation),
        ("模型验证", test_model_validation),
        ("组件功能", test_component_functionality),
        ("性能测试", test_performance),
    ]
    
    # 运行测试
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print_section(f"测试: {test_name}")
        try:
            if test_func():
                passed += 1
                print(f"\n? {test_name} 测试通过")
            else:
                print(f"\n? {test_name} 测试失败")
        except Exception as e:
            print(f"\n? {test_name} 测试异常: {e}")
    
    # 总结
    print_section("验证结果")
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("\n? 所有测试通过！")
        print("Decoder-only Transformer with MLA 实现完全正确！")
        print("\n? 项目特点:")
        print("  ? Pre-RMSNorm 预归一化结构")
        print("  ? MLA 多头潜在注意力机制")
        print("  ? RoPE 旋转位置编码")
        print("  ? SwiGLU 门控线性单元")
        print("  ? 完整的训练和推理流程")
        print("\n? 下一步:")
        print("  1. 运行 python examples/demo.py 查看功能演示")
        print("  2. 运行 python examples/train.py --model_size tiny 开始训练")
        print("  3. 查看 PROJECT_SUMMARY.md 了解详细信息")
    else:
        print(f"\n? {total - passed} 个测试失败")
        print("请检查实现并修复问题")
    
    print_section("验证完成")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
