"""
模型组件单元测试
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import torch
import torch.nn as nn
from transformers import AutoTokenizer

from src.config import ModelArgs, ModelConfigs
from src.utils import RMSNorm, precompute_freqs_cis, apply_rotary_emb
from src.attention import MLA, MultiHeadAttention
from src.feedforward import SwiGLU, TransformerBlock
from src.transformer import Transformer
from src.model_utils import validate_model_architecture


class TestUtils(unittest.TestCase):
    """测试工具函数"""
    
    def setUp(self):
        self.dim = 128
        self.seq_len = 32
        self.batch_size = 2
    
    def test_rms_norm(self):
        """测试 RMSNorm"""
        norm = RMSNorm(self.dim)
        x = torch.randn(self.batch_size, self.seq_len, self.dim)
        
        output = norm(x)
        
        # 检查输出形状
        self.assertEqual(output.shape, x.shape)
        
        # 检查 RMS 归一化效果
        rms = torch.sqrt(torch.mean(output**2, dim=-1, keepdim=True))
        # RMS 应该接近 1（考虑权重参数）
        self.assertTrue(torch.allclose(rms, torch.ones_like(rms), atol=0.1))
    
    def test_rope(self):
        """测试 RoPE"""
        head_dim = 64
        freqs_cis = precompute_freqs_cis(head_dim, self.seq_len)
        
        # 检查频率形状
        self.assertEqual(freqs_cis.shape, (self.seq_len, head_dim // 2))
        
        # 测试应用 RoPE
        q = torch.randn(self.batch_size, self.seq_len, 4, head_dim)
        k = torch.randn(self.batch_size, self.seq_len, 4, head_dim)
        
        q_rot, k_rot = apply_rotary_emb(q, k, freqs_cis)
        
        # 检查输出形状
        self.assertEqual(q_rot.shape, q.shape)
        self.assertEqual(k_rot.shape, k.shape)
        
        # 检查旋转不变性：||q|| = ||q_rot||
        q_norm = torch.norm(q, dim=-1)
        q_rot_norm = torch.norm(q_rot, dim=-1)
        self.assertTrue(torch.allclose(q_norm, q_rot_norm, atol=1e-5))


class TestAttention(unittest.TestCase):
    """测试注意力机制"""
    
    def setUp(self):
        self.args = ModelConfigs.tiny()
        self.batch_size = 2
        self.seq_len = 64
    
    def test_mla(self):
        """测试 MLA"""
        mla = MLA(self.args)
        x = torch.randn(self.batch_size, self.seq_len, self.args.d_model)
        freqs_cis = precompute_freqs_cis(self.args.qk_rope_head_dim, self.seq_len)
        
        output = mla(x, freqs_cis)
        
        # 检查输出形状
        self.assertEqual(output.shape, x.shape)
        
        # 检查梯度流
        loss = output.sum()
        loss.backward()
        
        # 检查是否有梯度
        for param in mla.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
    
    def test_standard_attention(self):
        """测试标准多头注意力"""
        attention = MultiHeadAttention(self.args)
        x = torch.randn(self.batch_size, self.seq_len, self.args.d_model)
        freqs_cis = precompute_freqs_cis(self.args.d_model // self.args.n_heads, self.seq_len)
        
        output = attention(x, freqs_cis)
        
        # 检查输出形状
        self.assertEqual(output.shape, x.shape)


class TestFeedForward(unittest.TestCase):
    """测试前馈网络"""
    
    def setUp(self):
        self.args = ModelConfigs.tiny()
        self.batch_size = 2
        self.seq_len = 64
    
    def test_swiglu(self):
        """测试 SwiGLU"""
        ffn = SwiGLU(self.args)
        x = torch.randn(self.batch_size, self.seq_len, self.args.d_model)
        
        output = ffn(x)
        
        # 检查输出形状
        self.assertEqual(output.shape, x.shape)
        
        # 检查梯度流
        loss = output.sum()
        loss.backward()
        
        for param in ffn.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
    
    def test_transformer_block(self):
        """测试 Transformer Block"""
        block = TransformerBlock(self.args)
        x = torch.randn(self.batch_size, self.seq_len, self.args.d_model)
        freqs_cis = precompute_freqs_cis(self.args.qk_rope_head_dim, self.seq_len)
        
        output = block(x, freqs_cis, None)
        
        # 检查输出形状
        self.assertEqual(output.shape, x.shape)
        
        # 检查残差连接：输出不应该等于输入
        self.assertFalse(torch.allclose(output, x))


class TestTransformer(unittest.TestCase):
    """测试完整的 Transformer 模型"""
    
    def setUp(self):
        self.args = ModelConfigs.tiny()
        self.batch_size = 2
        self.seq_len = 64
        self.model = Transformer(self.args)
    
    def test_forward_pass(self):
        """测试前向传播"""
        input_ids = torch.randint(0, self.args.vocab_size, (self.batch_size, self.seq_len))
        
        outputs = self.model(input_ids)
        logits = outputs["logits"]
        
        # 检查输出形状
        expected_shape = (self.batch_size, self.seq_len, self.args.vocab_size)
        self.assertEqual(logits.shape, expected_shape)
        
        # 检查 logits 是否为有限值
        self.assertTrue(torch.isfinite(logits).all())
    
    def test_loss_computation(self):
        """测试损失计算"""
        input_ids = torch.randint(0, self.args.vocab_size, (self.batch_size, self.seq_len))
        
        outputs = self.model(input_ids, labels=input_ids)
        loss = outputs["loss"]
        
        # 检查损失是否为标量
        self.assertEqual(loss.shape, ())
        
        # 检查损失是否为正数
        self.assertGreater(loss.item(), 0)
        
        # 检查损失是否为有限值
        self.assertTrue(torch.isfinite(loss))
    
    def test_generation(self):
        """测试文本生成"""
        self.model.eval()
        
        input_ids = torch.randint(0, self.args.vocab_size, (1, 10))
        
        with torch.no_grad():
            generated = self.model.generate(
                input_ids,
                max_new_tokens=20,
                do_sample=False  # 贪心解码
            )
        
        # 检查生成长度
        self.assertEqual(generated.shape[0], 1)
        self.assertGreater(generated.shape[1], input_ids.shape[1])
        
        # 检查生成的 token 是否在词汇表范围内
        self.assertTrue((generated >= 0).all())
        self.assertTrue((generated < self.args.vocab_size).all())
    
    def test_model_validation(self):
        """测试模型验证"""
        validation_results = validate_model_architecture(self.model, self.args)
        
        # 检查验证是否通过
        if not validation_results["passed"]:
            print("Validation errors:", validation_results["errors"])
            print("Validation warnings:", validation_results["warnings"])
        
        self.assertTrue(validation_results["passed"], 
                       f"Model validation failed: {validation_results['errors']}")
    
    def test_parameter_count(self):
        """测试参数数量"""
        from src.utils import count_parameters
        
        param_count = count_parameters(self.model)
        
        # 检查参数数量是否合理（对于 tiny 模型）
        self.assertGreater(param_count, 1000)  # 至少有一些参数
        self.assertLess(param_count, 10_000_000)  # 不应该太大
        
        print(f"Model has {param_count:,} parameters")


class TestTokenizer(unittest.TestCase):
    """测试 tokenizer 集成"""
    
    def setUp(self):
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_fast=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.args = ModelConfigs.tiny()
        self.args.vocab_size = len(self.tokenizer)
        self.model = Transformer(self.args)
    
    def test_tokenizer_integration(self):
        """测试 tokenizer 与模型的集成"""
        text = "Hello, world! This is a test."
        
        # 编码
        input_ids = self.tokenizer.encode(text, return_tensors="pt")
        
        # 检查 token IDs 是否在词汇表范围内
        self.assertTrue((input_ids >= 0).all())
        self.assertTrue((input_ids < self.args.vocab_size).all())
        
        # 模型前向传播
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs["logits"]
        
        # 检查输出形状
        expected_shape = (1, input_ids.shape[1], self.args.vocab_size)
        self.assertEqual(logits.shape, expected_shape)
        
        # 解码
        decoded = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        self.assertIsInstance(decoded, str)


def run_tests():
    """运行所有测试"""
    # 创建测试套件
    test_classes = [
        TestUtils,
        TestAttention, 
        TestFeedForward,
        TestTransformer,
        TestTokenizer
    ]
    
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    if success:
        print("\n? All tests passed!")
    else:
        print("\n? Some tests failed!")
        sys.exit(1)
