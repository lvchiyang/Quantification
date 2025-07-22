#!/usr/bin/env python3
"""
基本功能测试脚本
"""

import torch
import sys
import os

# 添加 src 目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """测试模块导入"""
    print("Testing imports...")
    
    try:
        from config import ModelArgs, ModelConfigs
        print("? Config imported successfully")
    except Exception as e:
        print(f"? Config import failed: {e}")
        return False
    
    try:
        from utils import RMSNorm, precompute_freqs_cis
        print("? Utils imported successfully")
    except Exception as e:
        print(f"? Utils import failed: {e}")
        return False
    
    try:
        from attention import MLA
        print("? Attention imported successfully")
    except Exception as e:
        print(f"? Attention import failed: {e}")
        return False
    
    try:
        from feedforward import SwiGLU
        print("? FeedForward imported successfully")
    except Exception as e:
        print(f"? FeedForward import failed: {e}")
        return False
    
    try:
        from transformer import Transformer
        print("? Transformer imported successfully")
    except Exception as e:
        print(f"? Transformer import failed: {e}")
        return False
    
    return True


def test_model_creation():
    """测试模型创建"""
    print("\nTesting model creation...")
    
    try:
        from config import ModelConfigs
        from transformer import Transformer
        
        # 创建 tiny 模型
        args = ModelConfigs.tiny()
        model = Transformer(args)
        
        print(f"? Model created successfully")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return True
    except Exception as e:
        print(f"? Model creation failed: {e}")
        return False


def test_forward_pass():
    """测试前向传播"""
    print("\nTesting forward pass...")
    
    try:
        from config import ModelConfigs
        from transformer import Transformer
        
        args = ModelConfigs.tiny()
        model = Transformer(args)
        
        # 创建随机输入
        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, args.vocab_size, (batch_size, seq_len))
        
        # 前向传播
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs["logits"]
        
        expected_shape = (batch_size, seq_len, args.vocab_size)
        if logits.shape == expected_shape:
            print(f"? Forward pass successful")
            print(f"  Input shape: {input_ids.shape}")
            print(f"  Output shape: {logits.shape}")
            return True
        else:
            print(f"? Wrong output shape: got {logits.shape}, expected {expected_shape}")
            return False
            
    except Exception as e:
        print(f"? Forward pass failed: {e}")
        return False


def test_loss_computation():
    """测试损失计算"""
    print("\nTesting loss computation...")
    
    try:
        from config import ModelConfigs
        from transformer import Transformer
        
        args = ModelConfigs.tiny()
        model = Transformer(args)
        
        # 创建随机输入
        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, args.vocab_size, (batch_size, seq_len))
        
        # 计算损失
        outputs = model(input_ids, labels=input_ids)
        loss = outputs["loss"]
        
        if loss is not None and torch.isfinite(loss):
            print(f"? Loss computation successful")
            print(f"  Loss value: {loss.item():.4f}")
            return True
        else:
            print(f"? Invalid loss: {loss}")
            return False
            
    except Exception as e:
        print(f"? Loss computation failed: {e}")
        return False


def test_generation():
    """测试文本生成"""
    print("\nTesting text generation...")
    
    try:
        from config import ModelConfigs
        from transformer import Transformer
        
        args = ModelConfigs.tiny()
        model = Transformer(args)
        model.eval()
        
        # 创建输入
        input_ids = torch.randint(0, args.vocab_size, (1, 10))
        
        # 生成文本
        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                max_new_tokens=20,
                do_sample=False
            )
        
        if generated.shape[1] > input_ids.shape[1]:
            print(f"? Text generation successful")
            print(f"  Input length: {input_ids.shape[1]}")
            print(f"  Generated length: {generated.shape[1]}")
            return True
        else:
            print(f"? No new tokens generated")
            return False
            
    except Exception as e:
        print(f"? Text generation failed: {e}")
        return False


def main():
    """运行所有测试"""
    print("=" * 60)
    print("Basic Functionality Tests")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_model_creation,
        test_forward_pass,
        test_loss_computation,
        test_generation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("? All tests passed! The model implementation is working correctly.")
    else:
        print("? Some tests failed. Please check the implementation.")
    
    print("=" * 60)
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
