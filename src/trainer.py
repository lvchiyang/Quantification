"""
训练器实现
包含训练循环、数据处理、模型保存等功能
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from datasets import load_dataset
import os
import json
import time
from typing import Optional, Dict, Any, List
from tqdm import tqdm
import matplotlib.pyplot as plt

from .config import ModelArgs
from .transformer import Transformer
from .utils import get_device


class TextDataset(Dataset):
    """文本数据集"""
    
    def __init__(
        self, 
        texts: List[str], 
        tokenizer, 
        max_length: int = 512,
        stride: int = 256
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        
        # 对所有文本进行 tokenize
        self.examples = []
        for text in tqdm(texts, desc="Tokenizing"):
            # 编码文本
            encoded = tokenizer.encode(text, add_special_tokens=True)
            
            # 滑动窗口切分长文本
            for i in range(0, len(encoded), stride):
                chunk = encoded[i:i + max_length]
                if len(chunk) >= 32:  # 过滤太短的序列
                    # 填充到固定长度
                    if len(chunk) < max_length:
                        chunk.extend([tokenizer.pad_token_id] * (max_length - len(chunk)))
                    self.examples.append(chunk)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return torch.tensor(self.examples[idx], dtype=torch.long)


class Trainer:
    """训练器"""
    
    def __init__(
        self,
        model: Transformer,
        args: ModelArgs,
        tokenizer,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        output_dir: str = "./checkpoints"
    ):
        self.model = model
        self.args = args
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.output_dir = output_dir
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 设备
        self.device = get_device()
        self.model.to(self.device)
        
        # 优化器
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=(args.beta1, args.beta2)
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=1000,  # 将在训练时更新
            eta_min=args.learning_rate * 0.1
        )
        
        # 训练状态
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        
        print(f"Trainer initialized on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(dataloader)
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # 数据移到设备
            input_ids = batch.to(self.device)
            
            # 前向传播
            outputs = self.model(input_ids=input_ids, labels=input_ids)
            loss = outputs["loss"]
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 优化器步骤
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            # 更新学习率
            self.scheduler.step()
            
            # 记录
            total_loss += loss.item()
            self.global_step += 1
            
            # 更新进度条
            current_lr = self.scheduler.get_last_lr()[0]
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss / (batch_idx + 1):.4f}',
                'lr': f'{current_lr:.2e}'
            })
            
            # 记录学习率
            if self.global_step % 10 == 0:
                self.learning_rates.append(current_lr)
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> float:
        """评估模型"""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(dataloader)
        
        progress_bar = tqdm(dataloader, desc="Evaluating")
        
        for batch in progress_bar:
            input_ids = batch.to(self.device)
            
            outputs = self.model(input_ids=input_ids, labels=input_ids)
            loss = outputs["loss"]
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def save_checkpoint(self, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'args': self.args,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates
        }
        
        # 保存最新检查点
        checkpoint_path = os.path.join(self.output_dir, 'latest_checkpoint.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳检查点
        if is_best:
            best_path = os.path.join(self.output_dir, 'best_checkpoint.pt')
            torch.save(checkpoint, best_path)
            print(f"New best model saved with validation loss: {self.best_val_loss:.4f}")
        
        # 保存模型配置
        config_path = os.path.join(self.output_dir, 'config.json')
        with open(config_path, 'w') as f:
            # 将 ModelArgs 转换为字典
            config_dict = {
                k: v for k, v in self.args.__dict__.items() 
                if not k.startswith('_')
            }
            json.dump(config_dict, f, indent=2)
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.learning_rates = checkpoint.get('learning_rates', [])
        
        print(f"Checkpoint loaded from {checkpoint_path}")
        print(f"Resuming from epoch {self.epoch}, step {self.global_step}")
    
    def train(
        self,
        num_epochs: int,
        batch_size: int = 16,
        eval_steps: int = 500,
        save_steps: int = 1000,
        max_eval_batches: Optional[int] = None
    ):
        """训练模型"""
        print(f"Starting training for {num_epochs} epochs...")
        
        # 创建数据加载器
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        val_dataloader = None
        if self.val_dataset is not None:
            val_dataloader = DataLoader(
                self.val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=True
            )
            if max_eval_batches is not None:
                # 限制验证批次数量
                val_dataloader = list(val_dataloader)[:max_eval_batches]
        
        # 更新学习率调度器
        total_steps = len(train_dataloader) * num_epochs
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=self.args.learning_rate * 0.1
        )
        
        # 训练循环
        for epoch in range(num_epochs):
            self.epoch = epoch
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # 训练
            train_loss = self.train_epoch(train_dataloader)
            print(f"Train loss: {train_loss:.4f}")
            
            # 验证
            if val_dataloader is not None:
                val_loss = self.evaluate(val_dataloader)
                self.val_losses.append(val_loss)
                print(f"Validation loss: {val_loss:.4f}")
                
                # 检查是否是最佳模型
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
            else:
                is_best = False
            
            # 保存检查点
            self.save_checkpoint(is_best=is_best)
            
            # 绘制训练曲线
            if epoch % 5 == 0 or epoch == num_epochs - 1:
                self.plot_training_curves()
        
        print("Training completed!")
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # 损失曲线
        axes[0].plot(self.train_losses, label='Train Loss')
        if self.val_losses:
            axes[0].plot(self.val_losses, label='Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # 学习率曲线
        if self.learning_rates:
            axes[1].plot(self.learning_rates)
            axes[1].set_xlabel('Step (x10)')
            axes[1].set_ylabel('Learning Rate')
            axes[1].set_title('Learning Rate Schedule')
            axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_curves.png'))
        plt.close()


def load_tiny_stories_dataset(tokenizer, max_samples: int = 10000):
    """加载 TinyStories 数据集"""
    print("Loading TinyStories dataset...")
    
    # 加载数据集
    dataset = load_dataset("roneneldan/TinyStories", split="train")
    
    # 限制样本数量
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    # 提取文本
    texts = [example["text"] for example in dataset]
    
    # 创建训练和验证集
    split_idx = int(0.9 * len(texts))
    train_texts = texts[:split_idx]
    val_texts = texts[split_idx:]
    
    # 创建数据集
    train_dataset = TextDataset(train_texts, tokenizer, max_length=512)
    val_dataset = TextDataset(val_texts, tokenizer, max_length=512)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    return train_dataset, val_dataset
