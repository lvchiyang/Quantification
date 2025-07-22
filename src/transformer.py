"""
完整的 Decoder-only Transformer 模型实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import math

from .config import ModelArgs
from .utils import RMSNorm, precompute_freqs_cis, create_causal_mask, count_parameters
from .feedforward import TransformerBlock


class Transformer(nn.Module):
    """
    Decoder-only Transformer 模型

    架构特点：
    - Pre-RMSNorm
    - MLA (Multi-Head Latent Attention)
    - RoPE (Rotary Position Embedding)
    - SwiGLU FFN
    - Next Token Prediction
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        # Token 嵌入
        self.tok_embed = nn.Embedding(args.vocab_size, args.d_model)

        # Transformer 层
        self.layers = nn.ModuleList([
            TransformerBlock(args, layer_idx=i)
            for i in range(args.n_layers)
        ])

        # 最终归一化层
        self.norm = RMSNorm(args.d_model, eps=args.layer_norm_eps)

        # 语言模型头
        if args.tie_word_embeddings:
            # 共享输入输出嵌入权重
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(args.d_model, args.vocab_size, bias=False)

        # 预计算 RoPE 频率
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(
                args.qk_rope_head_dim,
                args.max_seq_len * 2,  # 预留更长序列的空间
                args.rope_theta
            ),
            persistent=False
        )

        # 初始化权重
        self.apply(self._init_weights)

        # 打印模型信息
        self._print_model_info()

    def _init_weights(self, module: nn.Module):
        """初始化模型权重"""
        if isinstance(module, nn.Linear):
            # 使用标准正态分布初始化，标准差为 1/sqrt(fan_in)
            std = 1.0 / math.sqrt(module.in_features)
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _print_model_info(self):
        """打印模型信息"""
        num_params = count_parameters(self)
        print(f"\nTransformer Model Info:")
        print(f"  Total parameters: {num_params:,}")
        print(f"  Model size: ~{num_params * 4 / 1024**2:.1f} MB (fp32)")
        print(f"  Layers: {self.args.n_layers}")
        print(f"  Hidden size: {self.args.d_model}")
        print(f"  Attention heads: {self.args.n_heads}")
        print(f"  Vocab size: {self.args.vocab_size}")
        print(f"  Max sequence length: {self.args.max_seq_len}")

    def get_input_embeddings(self) -> nn.Embedding:
        """获取输入嵌入层"""
        return self.tok_embed

    def set_input_embeddings(self, embeddings: nn.Embedding):
        """设置输入嵌入层"""
        self.tok_embed = embeddings

    def get_output_embeddings(self) -> Optional[nn.Linear]:
        """获取输出嵌入层"""
        if self.args.tie_word_embeddings:
            return None
        return self.lm_head
    
    def set_output_embeddings(self, embeddings: Optional[nn.Linear]):
        """设置输出嵌入层"""
        if not self.args.tie_word_embeddings:
            self.lm_head = embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True
    ) -> Dict[str, Any]:
        """
        前向传播

        Args:
            input_ids: 输入 token IDs [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            labels: 标签 [batch_size, seq_len]，用于计算损失
            use_cache: 是否使用缓存（暂未实现）
            output_attentions: 是否输出注意力权重
            output_hidden_states: 是否输出隐藏状态
            return_dict: 是否返回字典格式

        Returns:
            模型输出
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # 1. Token 嵌入
        hidden_states = self.tok_embed(input_ids)  # [batch_size, seq_len, d_model]

        # 2. 获取 RoPE 频率
        freqs_cis = self.freqs_cis[:seq_len].to(device)

        # 3. 创建因果掩码
        causal_mask = None
        if attention_mask is not None:
            # 如果提供了 attention_mask，需要结合因果掩码
            # 这里简化处理，直接使用因果掩码
            pass

        # 4. 通过 Transformer 层
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            hidden_states = layer(
                hidden_states,
                freqs_cis=freqs_cis,
                attn_mask=causal_mask,
                is_causal=True
            )

        # 5. 最终归一化
        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 6. 语言模型头
        if self.args.tie_word_embeddings:
            # 使用输入嵌入的转置作为输出权重
            logits = F.linear(hidden_states, self.tok_embed.weight)
        else:
            logits = self.lm_head(hidden_states)

        # 7. 计算损失
        loss = None
        if labels is not None:
            # 移位标签：预测下一个 token
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # 计算交叉熵损失
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

        # 8. 返回结果
        if return_dict:
            return {
                "logits": logits,
                "loss": loss,
                "hidden_states": all_hidden_states,
                "attentions": all_attentions,
            }
        else:
            outputs = (logits,)
            if loss is not None:
                outputs = (loss,) + outputs
            if output_hidden_states:
                outputs = outputs + (all_hidden_states,)
            if output_attentions:
                outputs = outputs + (all_attentions,)
            return outputs
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None
    ) -> torch.Tensor:
        """
        生成文本

        Args:
            input_ids: 输入 token IDs [batch_size, seq_len]
            max_new_tokens: 最大生成 token 数
            temperature: 温度参数
            top_k: Top-K 采样
            top_p: Top-P 采样
            do_sample: 是否采样
            pad_token_id: 填充 token ID
            eos_token_id: 结束 token ID

        Returns:
            生成的 token IDs [batch_size, seq_len + max_new_tokens]
        """
        self.eval()

        batch_size = input_ids.shape[0]
        device = input_ids.device

        # 生成循环
        for _ in range(max_new_tokens):
            # 前向传播
            outputs = self.forward(input_ids, return_dict=True)
            logits = outputs["logits"]

            # 获取最后一个位置的 logits
            next_token_logits = logits[:, -1, :]  # [batch_size, vocab_size]

            # 应用温度
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # Top-K 采样
            if top_k is not None:
                top_k = min(top_k, next_token_logits.size(-1))
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                next_token_logits.scatter_(-1, top_k_indices, top_k_logits)

            # Top-P 采样
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # 移除累积概率超过 top_p 的 token
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')

            # 采样或贪心选择
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # 拼接新 token
            input_ids = torch.cat([input_ids, next_token], dim=-1)

            # 检查是否遇到结束 token
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

            # 检查序列长度限制
            if input_ids.shape[1] >= self.args.max_seq_len:
                break

        return input_ids
