# 金融特征嵌入层 - 特征分组说明

##  重要修改：成交次数移至成交量特征组

根据 `doc/data.md` 的要求，**成交次数**已从市场特征组移至成交量特征组。

###  修改后的特征分组（20维）

| 特征组 | 索引范围 | 维度 | 特征列表 | 说明 |
|--------|----------|------|----------|------|
| **时间特征** | [0-2] | 3维 | 月、日、星期 | 基础时间信息 |
| **价格特征** | [3-6] | 4维 | open_rel, high_rel, low_rel, close_rel | OHLC相对价格 |
| **成交量特征** | [7-11] | 5维 | volume_rel, volume_change, amount_rel, amount_change, **成交次数** | 交易活跃度指标 |
| **市场特征** | [12-15] | 4维 | 涨幅, 振幅, 换手%, **price_median** | 市场波动指标 |
| **金融特征** | [16-19] | 4维 | big_order_activity, chip_concentration, market_sentiment, price_volume_sync | 高级金融指标 |

###  主要变化

#### 变化前（旧版本）
```python
# 价格特征 (5维): price_median, open_rel, high_rel, low_rel, close_rel
# 成交量特征 (4维): volume_rel, volume_change, amount_rel, amount_change  
# 市场特征 (4维): 涨幅, 振幅, 换手%, 成交次数
```

#### 变化后（新版本）
```python
# 价格特征 (4维): open_rel, high_rel, low_rel, close_rel
# 成交量特征 (5维): volume_rel, volume_change, amount_rel, amount_change, 成交次数
# 市场特征 (4维): 涨幅, 振幅, 换手%, price_median
```

###  修改原因

1. **金融逻辑合理性**
   - 成交次数与成交量密切相关，都反映市场交易活跃度
   - 成交次数/成交量 = 平均每笔交易量，是重要的volume派生指标
   - 大单、小单的交易模式可以通过成交量和成交次数的关系体现

2. **数据处理一致性**
   - 成交次数需要标准化处理（不同股票量级差异大）
   - 与volume特征的处理方式一致
   - 都需要Batch序列内标准化

3. **特征工程优化**
   - price_median作为价格基准，更适合放在市场特征组
   - 成交量特征组现在包含完整的交易活跃度信息
   - 特征组内部语义更加统一

###  代码实现

```python
class FinancialEmbedding(nn.Module):
    def __init__(self, d_model=512):
        super().__init__()
        
        group_dim = d_model // 5
        
        # 时间特征 (3维)
        self.time_embed = nn.Linear(3, group_dim)
        
        # 价格特征 (4维) - 移除price_median
        self.price_embed = nn.Linear(4, group_dim)
        
        # 成交量特征 (5维) - 添加成交次数
        self.volume_embed = nn.Linear(5, group_dim)
        
        # 市场特征 (4维) - 添加price_median
        self.market_embed = nn.Linear(4, group_dim)
        
        # 金融特征 (4维)
        self.financial_embed = nn.Linear(4, group_dim)
    
    def forward(self, x):
        # 特征分组
        time_features = x[..., 0:3]       # [0-2]
        price_features = x[..., 3:7]      # [3-6]
        volume_features = x[..., 7:12]    # [7-11] 包含成交次数
        market_features = x[..., 12:16]   # [12-15] 包含price_median
        financial_features = x[..., 16:20] # [16-19]
        
        # 分组嵌入
        time_emb = self.time_embed(time_features)
        price_emb = self.price_embed(price_features)
        volume_emb = self.volume_embed(volume_features)
        market_emb = self.market_embed(market_features)
        financial_emb = self.financial_embed(financial_features)
        
        # 拼接
        return torch.cat([time_emb, price_emb, volume_emb, market_emb, financial_emb], dim=-1)
```

###  验证测试

运行以下测试验证修改：

```bash
cd src/price_prediction
python test_embedding_grouping.py
```

测试内容：
-  特征分组正确性
-  成交次数在volume组中的处理
-  维度分配正确性
-  不同交易模式的embedding效果

###  预期效果

1. **更好的特征表示**
   - 成交量特征组包含完整的交易活跃度信息
   - 特征组内部相关性更强，学习效果更好

2. **更合理的金融语义**
   - 符合金融分析的逻辑分组
   - 便于模型理解交易行为模式

3. **更稳定的训练**
   - 特征组内数值范围更统一
   - 减少不同特征组间的数值差异

###  相关文件

- `src/price_prediction/embedding.py` - 主要实现
- `src/price_prediction/test_embedding_grouping.py` - 验证测试
- `doc/data.md` - 数据处理文档（原始需求）

---

**注意**：这个修改影响所有使用该embedding层的模型，请确保相关代码同步更新！
