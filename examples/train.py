#!/usr/bin/env python3
"""
训练示例脚本
演示如何训练 Decoder-only Transformer 模型
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer
import argparse

from src.config import ModelArgs, ModelConfigs
from src.transformer import Transformer
from src.trainer import Trainer, load_tiny_stories_dataset


def main():
    parser = argparse.ArgumentParser(description="Train Decoder-only Transformer")
    parser.add_argument("--model_size", type=str, default="tiny", 
                       choices=["tiny", "small", "base", "large"],
                       help="Model size")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max_samples", type=int, default=10000, 
                       help="Maximum number of training samples")
    parser.add_argument("--output_dir", type=str, default="./checkpoints",
                       help="Output directory for checkpoints")
    parser.add_argument("--resume_from", type=str, default=None,
                       help="Resume training from checkpoint")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Training Decoder-only Transformer with MLA")
    print("=" * 60)
    
    # 1. 加载配置
    if args.model_size == "tiny":
        model_args = ModelConfigs.tiny()
    elif args.model_size == "small":
        model_args = ModelConfigs.small()
    elif args.model_size == "base":
        model_args = ModelConfigs.base()
    elif args.model_size == "large":
        model_args = ModelConfigs.large()
    else:
        raise ValueError(f"Unknown model size: {args.model_size}")
    
    # 更新学习率
    model_args.learning_rate = args.learning_rate
    
    print(f"Model configuration: {args.model_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of epochs: {args.num_epochs}")
    print(f"Max samples: {args.max_samples}")
    
    # 2. 加载 tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 更新词汇表大小
    model_args.vocab_size = len(tokenizer)
    print(f"Vocabulary size: {model_args.vocab_size}")
    
    # 3. 创建模型
    print("\nCreating model...")
    model = Transformer(model_args)
    
    # 4. 加载数据集
    print("\nLoading dataset...")
    train_dataset, val_dataset = load_tiny_stories_dataset(
        tokenizer, 
        max_samples=args.max_samples
    )
    
    # 5. 创建训练器
    print("\nCreating trainer...")
    trainer = Trainer(
        model=model,
        args=model_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=args.output_dir
    )
    
    # 6. 恢复训练（如果指定）
    if args.resume_from:
        print(f"\nResuming training from {args.resume_from}")
        trainer.load_checkpoint(args.resume_from)
    
    # 7. 开始训练
    print("\nStarting training...")
    trainer.train(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        eval_steps=100,
        save_steps=500
    )
    
    # 8. 测试生成
    print("\nTesting text generation...")
    test_generation(model, tokenizer)
    
    print("\nTraining completed successfully!")


def test_generation(model, tokenizer):
    """测试文本生成"""
    model.eval()
    
    prompts = [
        "Once upon a time",
        "The little girl",
        "In a magical forest"
    ]
    
    print("\nGenerated samples:")
    print("-" * 50)
    
    for prompt in prompts:
        # 编码输入
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
        
        # 生成文本
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=50,
                temperature=0.8,
                top_k=50,
                top_p=0.9,
                do_sample=True
            )
        
        # 解码输出
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        print(f"Prompt: {prompt}")
        print(f"Generated: {generated_text}")
        print("-" * 50)


if __name__ == "__main__":
    main()
