# 🔬 技术详情文档

本文档详细介绍金融量化交易系统的技术实现细节、架构设计和高级功能。

## 🏗️ 架构设计

### 两阶段解耦架构

```
阶段1: 价格预测网络训练
┌─────────────────────────────────┐
│ 金融数据 [180, 11]              │
│         ↓                       │
│ Transformer Encoder             │
│         ↓                       │
│ 价格预测头 → 未来7天价格         │
│ 特征提取头 → 策略特征 [d_model]  │
└─────────────────────────────────┘

阶段2: 策略网络训练
┌─────────────────────────────────┐
│ 策略特征 [20, d_model] (冻结)   │
│         ↓                       │
│ GRU策略网络 (20天递归)          │
│         ↓                       │
│ 仓位决策 → 0-10仓位             │
└─────────────────────────────────┘
```

### 梯度传播问题解决

#### 原问题分析
```python
# 原问题：argmax不可导，阻断梯度传播
expected_position = position_logits.argmax()  # ❌ 梯度阻断
gru_input = cat([features, expected_position])
strategy_state = gru_cell(gru_input, strategy_state)

# 实际梯度路径：
# final_loss → position_logits ✅
# position_logits → expected_position ❌ (argmax阻断)
# expected_position → gru_input ❌
# gru_input → strategy_state ❌
```

#### 解决方案
```python
# 价格网络：标准监督学习
price_loss = mse_loss(price_pred, price_target)
price_loss.backward()  # 梯度完整传播

# 策略网络：基于收益的强化学习
strategy_loss = -relative_return + risk_cost + opportunity_cost
strategy_loss.backward()  # 只优化策略参数
```

## 🧠 核心组件详解

### 1. MLA (Multi-Head Latent Attention)

#### 传统注意力 vs MLA
```python
# 传统注意力
Traditional: Q@K^T@V  # O(n²d) 复杂度

# MLA
MLA: Q@(W_kv@X)^T@(W_kv@X)  # O(nd²) 复杂度，d<<n时更高效

# K/V压缩示例
original_kv_dim = 1024    # 原始维度
compressed_dim = 256      # 压缩后维度
compression_ratio = 4     # 压缩比例
```

#### MLA实现细节
```python
class MLA(nn.Module):
    def __init__(self, args):
        # K/V 潜在投影压缩
        self.kv_compress = nn.Linear(args.d_model, self.kv_lora_rank, bias=False)
        self.kv_norm = RMSNorm(self.kv_lora_rank, eps=args.layer_norm_eps)
        
        # 从压缩表示恢复 K/V
        self.k_up = nn.Linear(self.kv_lora_rank, self.n_heads * self.qk_head_dim, bias=False)
        self.v_up = nn.Linear(self.kv_lora_rank, self.n_heads * self.v_dim, bias=False)
        
        # 查询投影，分为 RoPE 和非 RoPE 部分
        self.q_nope = nn.Linear(args.d_model, self.n_heads * self.qk_nope_dim, bias=False)
        self.q_rope = nn.Linear(args.d_model, self.n_heads * self.qk_rope_dim, bias=False)
```

### 2. RoPE (Rotary Position Embedding)

#### 位置编码原理
```python
def precompute_freqs_cis(dim: int, end: int, theta: float = 1e4):
    # 计算频率
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    
    # 生成位置索引
    t = torch.arange(end, device=freqs.device, dtype=freqs.dtype)
    
    # 计算每个位置和频率的外积
    freqs = torch.outer(t, freqs).float()
    
    # 转换为复数形式 e^(i*theta) = cos(theta) + i*sin(theta)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    
    return freqs_cis
```

### 3. SwiGLU 前馈网络

#### 激活函数对比
```python
# 标准FFN
FFN(x) = W2 * ReLU(W1 * x + b1) + b2

# SwiGLU
SwiGLU(x) = (Swish(W1 * x) ⊙ W3 * x) * W2
# 其中 Swish(x) = x * sigmoid(x) = SiLU(x)
# ⊙ 表示逐元素相乘

# GeGLU (替代方案)
GeGLU(x) = (GELU(W1 * x) ⊙ W3 * x) * W2
```

### 4. GRU策略网络

#### 递归状态更新机制
```python
# 梯度传播路径
final_loss → position_logits[19] → strategy_state[19]
          → position_logits[18] → strategy_state[18]
          → ... → position_logits[0]

# 内存优化
for day in range(20):
    # 保持梯度的部分
    position = model.predict_position(features[day], strategy_state)
    strategy_state = model.update_state(strategy_state, position)

    # 不保持梯度的部分
    with torch.no_grad():
        portfolio_value *= (1 + position * returns[day])
```

#### 可微分仓位预测
```python
class GumbelSoftmaxPositionHead(nn.Module):
    def forward(self, x, temperature=1.0, hard=False):
        logits = self.linear(x)  # [batch_size, 11] (0-10仓位)
        
        if self.training:
            # 训练时：Gumbel-Softmax，可微分
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits)))
            y = F.softmax((logits + gumbel_noise) / temperature, dim=-1)
            
            if hard:
                # 直通估计器
                y_hard = torch.zeros_like(y)
                y_hard.scatter_(1, y.argmax(dim=1, keepdim=True), 1.0)
                y = (y_hard - y).detach() + y
                
            # 计算期望仓位
            position_values = torch.arange(11, device=x.device, dtype=torch.float32)
            positions = torch.sum(y * position_values, dim=-1, keepdim=True)
        else:
            # 推理时：直接argmax，离散
            positions = logits.argmax(dim=-1, keepdim=True).float()
            
        return {
            'logits': logits,
            'positions': positions,
            'probabilities': F.softmax(logits, dim=-1)
        }
```

## 📊 损失函数设计

### 1. 价格预测损失
```python
class PricePredictionLoss(nn.Module):
    def __init__(self, loss_type='mse'):
        self.loss_type = loss_type
        
    def forward(self, predictions, targets):
        if self.loss_type == 'mse':
            return F.mse_loss(predictions, targets)
        elif self.loss_type == 'mae':
            return F.l1_loss(predictions, targets)
        elif self.loss_type == 'huber':
            return F.huber_loss(predictions, targets)
```

### 2. 策略损失函数
```python
class StrategyLoss(nn.Module):
    def forward(self, position_predictions, next_day_returns):
        total_loss = 0
        
        for b in range(batch_size):
            positions = position_predictions[b, :, 0]  # [seq_len]
            returns = next_day_returns[b, :]           # [seq_len]
            
            # 1. 判断市场状态
            market_type = self.market_classifier.classify_market(returns)
            
            # 2. 计算相对基准收益
            relative_return_loss = self._calculate_relative_return_loss(
                positions, returns, market_type
            )
            
            # 3. 计算风险成本
            risk_cost = self._calculate_risk_cost(positions, returns)
            
            # 4. 计算机会成本
            opportunity_cost = self._calculate_opportunity_cost(
                positions, returns, market_type
            )
            
            # 5. 综合损失
            sample_loss = (
                self.relative_return_weight * relative_return_loss +
                self.risk_cost_weight * risk_cost +
                self.opportunity_cost_weight * opportunity_cost
            )
            
            total_loss += sample_loss
            
        return total_loss / batch_size
```

### 3. 市场分类算法
```python
def classify_market(self, returns):
    # 1. 统计特征
    mean_return = torch.mean(returns)
    volatility = torch.std(returns)

    # 2. 技术指标
    ma_short = torch.mean(returns[-5:])
    ma_long = torch.mean(returns[-10:])

    # 3. 投票机制
    votes = [
        self._simple_classifier(mean_return),
        self._technical_classifier(ma_short, ma_long),
        self._adaptive_classifier(returns)
    ]

    return self._majority_vote(votes)
```

## 🎯 性能优化

### 训练性能对比
| 配置 | 参数量 | 训练速度 | 内存使用 | 推荐场景 |
|------|--------|----------|----------|----------|
| Tiny | 2.5M | 快 | 2GB | 快速实验 |
| Small | 10M | 中等 | 4GB | 个人开发 |
| Base | 40M | 慢 | 8GB | 服务器训练 |
| Large | 160M | 很慢 | 16GB | 生产环境 |

### 内存优化技巧
```python
# 1. 梯度累积
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# 2. 混合精度训练
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## 🛠️ 故障排除

### 常见问题

**Q: 训练时内存不足？**
```python
# 解决方案：减小批次大小
config.batch_size = 1  # 从默认的4减少到1
config.strategy_state_dim = 64  # 减小状态维度
```

**Q: 梯度消失或爆炸？**
```python
# 解决方案：调整学习率和梯度裁剪
optimizer = optim.AdamW(model.parameters(), lr=1e-4)  # 降低学习率
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
```

**Q: 模型不收敛？**
```python
# 解决方案：检查数据和损失权重
config.information_ratio_weight = 0.5  # 降低信息比率权重
config.opportunity_cost_weight = 0.05   # 降低机会成本权重
```

### 调试技巧
```python
# 1. 检查梯度流
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.norm().item():.6f}")

# 2. 监控状态变化
print(f"状态变化: {torch.mean(torch.abs(new_state - old_state)).item():.6f}")

# 3. 验证市场分类
market_type = model.market_classifier.classify_market(returns)
print(f"市场类型: {market_type}")
```

## 📈 扩展功能

### 1. 专家混合网络 (MoE)
```python
class MoEFFN(nn.Module):
    def __init__(self, args, num_experts=8, top_k=2):
        self.gate = nn.Linear(args.d_model, num_experts, bias=False)
        self.experts = nn.ModuleList([
            SwiGLU(args) for _ in range(num_experts)
        ])
```

### 2. 多资产组合优化
```python
# 未来功能：支持多只股票的组合优化
class PortfolioOptimizer(nn.Module):
    def __init__(self, num_assets, d_model):
        self.asset_encoders = nn.ModuleList([
            PriceTransformer(config) for _ in range(num_assets)
        ])
        self.portfolio_head = nn.Linear(d_model * num_assets, num_assets)
```

### 3. 强化学习集成
```python
# 未来功能：集成PPO/SAC等强化学习算法
class RLStrategyNetwork(nn.Module):
    def __init__(self, config):
        self.actor = GRUStrategyNetwork(config)
        self.critic = nn.Linear(config.d_model, 1)
```

## 🔮 未来规划

### 即将推出的功能
- [ ] **多资产组合**: 支持多只股票的组合优化
- [ ] **强化学习**: 集成PPO/SAC等强化学习算法
- [ ] **实时交易**: 对接实盘交易接口
- [ ] **风险模型**: 更精细的风险控制模块
- [ ] **因子挖掘**: 自动特征工程和因子发现

### 技术改进
- [ ] **模型压缩**: 知识蒸馏和模型剪枝
- [ ] **分布式训练**: 多GPU和多机训练支持
- [ ] **在线学习**: 增量学习和模型更新
- [ ] **解释性**: 注意力可视化和决策解释

---

**📖 本文档持续更新中，欢迎提出建议和问题！**
