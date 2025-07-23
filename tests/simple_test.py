"""
简单测试脚本 - 验证模型实现
"""

# 测试基本导入
try:
    import torch
    print("PyTorch imported")
except ImportError as e:
    print(f"PyTorch import failed: {e}")
    exit(1)

try:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
    
    from config import ModelArgs, ModelConfigs
    print("Config imported")
    
    from utils import RMSNorm, precompute_freqs_cis
    print("Utils imported")
    
    from attention import MLA
    print("Attention imported")
    
    from feedforward import SwiGLU
    print("FeedForward imported")
    
    from transformer import Transformer
    print("Transformer imported")
    
except ImportError as e:
    print(f"Import failed: {e}")
    exit(1)

# 测试模型创建
try:
    print("\nTesting model creation...")
    args = ModelConfigs.tiny()
    model = Transformer(args)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
except Exception as e:
    print(f"Model creation failed: {e}")
    exit(1)

# 测试前向传播
try:
    print("\nTesting forward pass...")
    input_ids = torch.randint(0, args.vocab_size, (2, 32))
    
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs["logits"]
    
    print(f"Forward pass successful: {input_ids.shape} -> {logits.shape}")
except Exception as e:
    print(f"Forward pass failed: {e}")
    exit(1)

# 测试损失计算
try:
    print("\nTesting loss computation...")
    outputs = model(input_ids, labels=input_ids)
    loss = outputs["loss"]
    print(f"Loss computation successful: {loss.item():.4f}")
except Exception as e:
    print(f"Loss computation failed: {e}")
    exit(1)

# 测试生成
try:
    print("\nTesting generation...")
    model.eval()
    input_ids = torch.randint(0, args.vocab_size, (1, 10))
    
    with torch.no_grad():
        generated = model.generate(input_ids, max_new_tokens=10, do_sample=False)
    
    print(f"Generation successful: {input_ids.shape[1]} -> {generated.shape[1]} tokens")
except Exception as e:
    print(f"Generation failed: {e}")
    exit(1)

print("\nAll tests passed! The Decoder-only Transformer with MLA is working correctly!")
print("\nModel Summary:")
print(f"  Architecture: Decoder-only Transformer")
print(f"  Attention: Multi-Head Latent Attention (MLA)")
print(f"  Position Encoding: RoPE")
print(f"  Feed Forward: SwiGLU")
print(f"  Normalization: Pre-RMSNorm")
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"  Model size: ~{sum(p.numel() for p in model.parameters()) * 4 / 1024**2:.1f} MB (fp32)")
