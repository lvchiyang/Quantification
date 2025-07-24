#!/usr/bin/env python3
"""
文本生成示例脚本
演示如何使用训练好的模型生成文本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer
import argparse
import json

from src.config import ModelArgs
from src.price_prediction.transformer import Transformer


def load_model_from_checkpoint(checkpoint_path: str, device: str = "auto"):
    """从检查点加载模型"""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 加载配置
    model_args = checkpoint['args']
    print(f"Model configuration loaded:")
    print(f"  d_model: {model_args.d_model}")
    print(f"  n_layers: {model_args.n_layers}")
    print(f"  n_heads: {model_args.n_heads}")
    print(f"  vocab_size: {model_args.vocab_size}")
    
    # 创建模型
    model = Transformer(model_args)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded on device: {device}")
    return model, model_args


def interactive_generation(model, tokenizer, device):
    """交互式文本生成"""
    print("\n" + "=" * 60)
    print("Interactive Text Generation")
    print("Type 'quit' to exit, 'help' for commands")
    print("=" * 60)
    
    # 默认生成参数
    generation_config = {
        "max_new_tokens": 100,
        "temperature": 0.8,
        "top_k": 50,
        "top_p": 0.9,
        "do_sample": True
    }
    
    while True:
        try:
            # 获取用户输入
            prompt = input("\nEnter prompt: ").strip()
            
            if prompt.lower() == 'quit':
                break
            elif prompt.lower() == 'help':
                print("\nCommands:")
                print("  quit - Exit the program")
                print("  help - Show this help message")
                print("  config - Show/modify generation parameters")
                print("\nGeneration parameters:")
                for key, value in generation_config.items():
                    print(f"  {key}: {value}")
                continue
            elif prompt.lower() == 'config':
                print("\nCurrent generation parameters:")
                for key, value in generation_config.items():
                    print(f"  {key}: {value}")
                
                # 允许用户修改参数
                param = input("\nEnter parameter to modify (or press Enter to skip): ").strip()
                if param in generation_config:
                    try:
                        if param in ["max_new_tokens", "top_k"]:
                            new_value = int(input(f"Enter new value for {param}: "))
                        elif param in ["temperature", "top_p"]:
                            new_value = float(input(f"Enter new value for {param}: "))
                        elif param == "do_sample":
                            new_value = input(f"Enter new value for {param} (true/false): ").lower() == 'true'
                        else:
                            continue
                        
                        generation_config[param] = new_value
                        print(f"Updated {param} to {new_value}")
                    except ValueError:
                        print("Invalid value entered")
                continue
            
            if not prompt:
                continue
            
            # 生成文本
            print(f"\nGenerating text for prompt: '{prompt}'")
            print("-" * 40)
            
            generated_text = generate_text(
                model, tokenizer, prompt, device, **generation_config
            )
            
            print(f"Generated text:\n{generated_text}")
            print("-" * 40)
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


def generate_text(model, tokenizer, prompt, device, **kwargs):
    """生成文本"""
    # 编码输入
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # 生成
    with torch.no_grad():
        generated_ids = model.generate(input_ids=input_ids, **kwargs)
    
    # 解码
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    return generated_text


def batch_generation(model, tokenizer, prompts, device, output_file=None, **kwargs):
    """批量生成文本"""
    results = []
    
    print(f"\nGenerating text for {len(prompts)} prompts...")
    
    for i, prompt in enumerate(prompts):
        print(f"Processing prompt {i+1}/{len(prompts)}: '{prompt[:50]}...'")
        
        try:
            generated_text = generate_text(model, tokenizer, prompt, device, **kwargs)
            
            result = {
                "prompt": prompt,
                "generated": generated_text,
                "parameters": kwargs
            }
            results.append(result)
            
            print(f"Generated: {generated_text[:100]}...")
            
        except Exception as e:
            print(f"Error generating for prompt '{prompt}': {e}")
            results.append({
                "prompt": prompt,
                "generated": None,
                "error": str(e)
            })
    
    # 保存结果
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {output_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Generate text with trained model")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--mode", type=str, default="interactive",
                       choices=["interactive", "batch"],
                       help="Generation mode")
    parser.add_argument("--prompts", type=str, nargs="+",
                       help="Prompts for batch generation")
    parser.add_argument("--prompts_file", type=str,
                       help="File containing prompts (one per line)")
    parser.add_argument("--output_file", type=str,
                       help="Output file for batch generation results")
    parser.add_argument("--max_new_tokens", type=int, default=100,
                       help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                       help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50,
                       help="Top-k sampling")
    parser.add_argument("--top_p", type=float, default=0.9,
                       help="Top-p sampling")
    parser.add_argument("--no_sample", action="store_true",
                       help="Use greedy decoding instead of sampling")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cpu, cuda)")
    
    args = parser.parse_args()
    
    # 加载 tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    model, model_args = load_model_from_checkpoint(args.checkpoint, args.device)
    device = next(model.parameters()).device
    
    # 生成参数
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "do_sample": not args.no_sample
    }
    
    if args.mode == "interactive":
        # 交互式生成
        interactive_generation(model, tokenizer, device)
    
    elif args.mode == "batch":
        # 批量生成
        prompts = []
        
        if args.prompts:
            prompts.extend(args.prompts)
        
        if args.prompts_file:
            with open(args.prompts_file, 'r', encoding='utf-8') as f:
                prompts.extend([line.strip() for line in f if line.strip()])
        
        if not prompts:
            # 默认提示
            prompts = [
                "Once upon a time",
                "The little girl walked into the forest and",
                "In a magical kingdom far away",
                "The brave knight decided to",
                "On a sunny morning, the cat"
            ]
        
        batch_generation(
            model, tokenizer, prompts, device, 
            output_file=args.output_file,
            **generation_kwargs
        )


if __name__ == "__main__":
    main()
