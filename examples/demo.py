#!/usr/bin/env python3
"""
��ʾ�ű�
չʾ Decoder-only Transformer ģ�͵Ļ�������
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer
import argparse

from src.config import ModelArgs, ModelConfigs
from src.transformer import Transformer
from src.model_utils import validate_model_architecture, save_model_info
from src.utils import count_parameters


def demo_model_creation():
    """��ʾģ�ʹ���"""
    print("=" * 60)
    print("1. Model Creation Demo")
    print("=" * 60)
    
    # ������ͬ��С��ģ��
    configs = {
        "tiny": ModelConfigs.tiny(),
        "small": ModelConfigs.small(),
        "base": ModelConfigs.base()
    }
    
    for name, args in configs.items():
        print(f"\n{name.upper()} Model:")
        model = Transformer(args)
        param_count = count_parameters(model)
        print(f"  Parameters: {param_count:,}")
        print(f"  Model size: ~{param_count * 4 / 1024**2:.1f} MB (fp32)")


def demo_forward_pass():
    """��ʾǰ�򴫲�"""
    print("\n" + "=" * 60)
    print("2. Forward Pass Demo")
    print("=" * 60)
    
    # ʹ�� tiny ģ�ͽ�����ʾ
    args = ModelConfigs.tiny()
    model = Transformer(args)
    
    # �����������
    batch_size, seq_len = 2, 32
    input_ids = torch.randint(0, args.vocab_size, (batch_size, seq_len))
    
    print(f"Input shape: {input_ids.shape}")
    
    # ǰ�򴫲�
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs["logits"]
    
    print(f"Output logits shape: {logits.shape}")
    print(f"Logits range: [{logits.min():.3f}, {logits.max():.3f}]")
    
    # ������ʧ
    outputs_with_loss = model(input_ids, labels=input_ids)
    loss = outputs_with_loss["loss"]
    print(f"Loss: {loss.item():.4f}")


def demo_text_generation():
    """��ʾ�ı�����"""
    print("\n" + "=" * 60)
    print("3. Text Generation Demo")
    print("=" * 60)
    
    # ���� tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # ����ģ��
    args = ModelConfigs.tiny()
    args.vocab_size = len(tokenizer)
    model = Transformer(args)
    model.eval()
    
    # ������ʾ
    prompts = [
        "Once upon a time",
        "The quick brown fox",
        "In a galaxy far, far away"
    ]
    
    print(f"Model vocabulary size: {args.vocab_size}")
    print("\nGenerating text (note: model is untrained, output will be random):")
    print("-" * 50)
    
    for prompt in prompts:
        # ��������
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        
        # �����ı�
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=20,
                temperature=1.0,
                do_sample=True
            )
        
        # �������
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        print(f"Prompt: '{prompt}'")
        print(f"Generated: '{generated_text}'")
        print("-" * 50)


def demo_model_validation():
    """��ʾģ����֤"""
    print("\n" + "=" * 60)
    print("4. Model Validation Demo")
    print("=" * 60)
    
    args = ModelConfigs.tiny()
    model = Transformer(args)
    
    print("Running model validation...")
    validation_results = validate_model_architecture(model, args)
    
    print(f"Validation passed: {validation_results['passed']}")
    
    if validation_results['errors']:
        print("Errors:")
        for error in validation_results['errors']:
            print(f"  - {error}")
    
    if validation_results['warnings']:
        print("Warnings:")
        for warning in validation_results['warnings']:
            print(f"  - {warning}")
    
    print("Model info:")
    for key, value in validation_results['info'].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


def demo_architecture_comparison():
    """��ʾ��ͬ�ܹ�����ıȽ�"""
    print("\n" + "=" * 60)
    print("5. Architecture Components Demo")
    print("=" * 60)
    
    from src.attention import MLA, MultiHeadAttention
    from src.feedforward import SwiGLU
    
    args = ModelConfigs.tiny()
    batch_size, seq_len = 2, 32
    x = torch.randn(batch_size, seq_len, args.d_model)
    
    # MLA vs ��׼ע����
    print("Comparing attention mechanisms:")
    
    mla = MLA(args)
    std_attn = MultiHeadAttention(args)
    
    mla_params = count_parameters(mla)
    std_params = count_parameters(std_attn)
    
    print(f"  MLA parameters: {mla_params:,}")
    print(f"  Standard attention parameters: {std_params:,}")
    print(f"  Parameter ratio (MLA/Std): {mla_params/std_params:.2f}")
    
    # SwiGLU
    print("\nSwiGLU FFN:")
    ffn = SwiGLU(args)
    ffn_params = count_parameters(ffn)
    print(f"  SwiGLU parameters: {ffn_params:,}")
    
    # ����ǰ�򴫲�ʱ��
    import time
    
    print("\nForward pass timing (CPU):")
    
    # MLA
    start_time = time.time()
    for _ in range(100):
        with torch.no_grad():
            _ = mla(x, torch.randn(seq_len, args.qk_rope_head_dim // 2, dtype=torch.complex64))
    mla_time = time.time() - start_time
    
    # ��׼ע����
    start_time = time.time()
    for _ in range(100):
        with torch.no_grad():
            _ = std_attn(x, torch.randn(seq_len, args.d_model // args.n_heads // 2, dtype=torch.complex64))
    std_time = time.time() - start_time
    
    print(f"  MLA: {mla_time*10:.2f}ms per forward pass")
    print(f"  Standard: {std_time*10:.2f}ms per forward pass")
    print(f"  Speed ratio (MLA/Std): {mla_time/std_time:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Transformer Model Demo")
    parser.add_argument("--demo", type=str, default="all",
                       choices=["all", "creation", "forward", "generation", "validation", "comparison"],
                       help="Which demo to run")
    parser.add_argument("--save_info", type=str, default=None,
                       help="Directory to save model info")
    
    args = parser.parse_args()
    
    print("Decoder-only Transformer with MLA - Demo")
    print("This demo showcases the model architecture and capabilities")
    
    if args.demo in ["all", "creation"]:
        demo_model_creation()
    
    if args.demo in ["all", "forward"]:
        demo_forward_pass()
    
    if args.demo in ["all", "generation"]:
        demo_text_generation()
    
    if args.demo in ["all", "validation"]:
        demo_model_validation()
    
    if args.demo in ["all", "comparison"]:
        demo_architecture_comparison()
    
    # ����ģ����Ϣ
    if args.save_info:
        print(f"\n" + "=" * 60)
        print("Saving Model Information")
        print("=" * 60)
        
        model_args = ModelConfigs.tiny()
        model = Transformer(model_args)
        save_model_info(model, model_args, args.save_info)
    
    print(f"\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run tests: python tests/test_model.py")
    print("2. Train model: python examples/train.py --model_size tiny --num_epochs 1")
    print("3. Generate text: python examples/generate.py --checkpoint ./checkpoints/best_checkpoint.pt")


if __name__ == "__main__":
    main()
