"""
ģ�������Ԫ����
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
    """���Թ��ߺ���"""
    
    def setUp(self):
        self.dim = 128
        self.seq_len = 32
        self.batch_size = 2
    
    def test_rms_norm(self):
        """���� RMSNorm"""
        norm = RMSNorm(self.dim)
        x = torch.randn(self.batch_size, self.seq_len, self.dim)
        
        output = norm(x)
        
        # ��������״
        self.assertEqual(output.shape, x.shape)
        
        # ��� RMS ��һ��Ч��
        rms = torch.sqrt(torch.mean(output**2, dim=-1, keepdim=True))
        # RMS Ӧ�ýӽ� 1������Ȩ�ز�����
        self.assertTrue(torch.allclose(rms, torch.ones_like(rms), atol=0.1))
    
    def test_rope(self):
        """���� RoPE"""
        head_dim = 64
        freqs_cis = precompute_freqs_cis(head_dim, self.seq_len)
        
        # ���Ƶ����״
        self.assertEqual(freqs_cis.shape, (self.seq_len, head_dim // 2))
        
        # ����Ӧ�� RoPE
        q = torch.randn(self.batch_size, self.seq_len, 4, head_dim)
        k = torch.randn(self.batch_size, self.seq_len, 4, head_dim)
        
        q_rot, k_rot = apply_rotary_emb(q, k, freqs_cis)
        
        # ��������״
        self.assertEqual(q_rot.shape, q.shape)
        self.assertEqual(k_rot.shape, k.shape)
        
        # �����ת�����ԣ�||q|| = ||q_rot||
        q_norm = torch.norm(q, dim=-1)
        q_rot_norm = torch.norm(q_rot, dim=-1)
        self.assertTrue(torch.allclose(q_norm, q_rot_norm, atol=1e-5))


class TestAttention(unittest.TestCase):
    """����ע��������"""
    
    def setUp(self):
        self.args = ModelConfigs.tiny()
        self.batch_size = 2
        self.seq_len = 64
    
    def test_mla(self):
        """���� MLA"""
        mla = MLA(self.args)
        x = torch.randn(self.batch_size, self.seq_len, self.args.d_model)
        freqs_cis = precompute_freqs_cis(self.args.qk_rope_head_dim, self.seq_len)
        
        output = mla(x, freqs_cis)
        
        # ��������״
        self.assertEqual(output.shape, x.shape)
        
        # ����ݶ���
        loss = output.sum()
        loss.backward()
        
        # ����Ƿ����ݶ�
        for param in mla.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
    
    def test_standard_attention(self):
        """���Ա�׼��ͷע����"""
        attention = MultiHeadAttention(self.args)
        x = torch.randn(self.batch_size, self.seq_len, self.args.d_model)
        freqs_cis = precompute_freqs_cis(self.args.d_model // self.args.n_heads, self.seq_len)
        
        output = attention(x, freqs_cis)
        
        # ��������״
        self.assertEqual(output.shape, x.shape)


class TestFeedForward(unittest.TestCase):
    """����ǰ������"""
    
    def setUp(self):
        self.args = ModelConfigs.tiny()
        self.batch_size = 2
        self.seq_len = 64
    
    def test_swiglu(self):
        """���� SwiGLU"""
        ffn = SwiGLU(self.args)
        x = torch.randn(self.batch_size, self.seq_len, self.args.d_model)
        
        output = ffn(x)
        
        # ��������״
        self.assertEqual(output.shape, x.shape)
        
        # ����ݶ���
        loss = output.sum()
        loss.backward()
        
        for param in ffn.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
    
    def test_transformer_block(self):
        """���� Transformer Block"""
        block = TransformerBlock(self.args)
        x = torch.randn(self.batch_size, self.seq_len, self.args.d_model)
        freqs_cis = precompute_freqs_cis(self.args.qk_rope_head_dim, self.seq_len)
        
        output = block(x, freqs_cis, None)
        
        # ��������״
        self.assertEqual(output.shape, x.shape)
        
        # ���в����ӣ������Ӧ�õ�������
        self.assertFalse(torch.allclose(output, x))


class TestTransformer(unittest.TestCase):
    """���������� Transformer ģ��"""
    
    def setUp(self):
        self.args = ModelConfigs.tiny()
        self.batch_size = 2
        self.seq_len = 64
        self.model = Transformer(self.args)
    
    def test_forward_pass(self):
        """����ǰ�򴫲�"""
        input_ids = torch.randint(0, self.args.vocab_size, (self.batch_size, self.seq_len))
        
        outputs = self.model(input_ids)
        logits = outputs["logits"]
        
        # ��������״
        expected_shape = (self.batch_size, self.seq_len, self.args.vocab_size)
        self.assertEqual(logits.shape, expected_shape)
        
        # ��� logits �Ƿ�Ϊ����ֵ
        self.assertTrue(torch.isfinite(logits).all())
    
    def test_loss_computation(self):
        """������ʧ����"""
        input_ids = torch.randint(0, self.args.vocab_size, (self.batch_size, self.seq_len))
        
        outputs = self.model(input_ids, labels=input_ids)
        loss = outputs["loss"]
        
        # �����ʧ�Ƿ�Ϊ����
        self.assertEqual(loss.shape, ())
        
        # �����ʧ�Ƿ�Ϊ����
        self.assertGreater(loss.item(), 0)
        
        # �����ʧ�Ƿ�Ϊ����ֵ
        self.assertTrue(torch.isfinite(loss))
    
    def test_generation(self):
        """�����ı�����"""
        self.model.eval()
        
        input_ids = torch.randint(0, self.args.vocab_size, (1, 10))
        
        with torch.no_grad():
            generated = self.model.generate(
                input_ids,
                max_new_tokens=20,
                do_sample=False  # ̰�Ľ���
            )
        
        # ������ɳ���
        self.assertEqual(generated.shape[0], 1)
        self.assertGreater(generated.shape[1], input_ids.shape[1])
        
        # ������ɵ� token �Ƿ��ڴʻ��Χ��
        self.assertTrue((generated >= 0).all())
        self.assertTrue((generated < self.args.vocab_size).all())
    
    def test_model_validation(self):
        """����ģ����֤"""
        validation_results = validate_model_architecture(self.model, self.args)
        
        # �����֤�Ƿ�ͨ��
        if not validation_results["passed"]:
            print("Validation errors:", validation_results["errors"])
            print("Validation warnings:", validation_results["warnings"])
        
        self.assertTrue(validation_results["passed"], 
                       f"Model validation failed: {validation_results['errors']}")
    
    def test_parameter_count(self):
        """���Բ�������"""
        from src.utils import count_parameters
        
        param_count = count_parameters(self.model)
        
        # �����������Ƿ�������� tiny ģ�ͣ�
        self.assertGreater(param_count, 1000)  # ������һЩ����
        self.assertLess(param_count, 10_000_000)  # ��Ӧ��̫��
        
        print(f"Model has {param_count:,} parameters")


class TestTokenizer(unittest.TestCase):
    """���� tokenizer ����"""
    
    def setUp(self):
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_fast=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.args = ModelConfigs.tiny()
        self.args.vocab_size = len(self.tokenizer)
        self.model = Transformer(self.args)
    
    def test_tokenizer_integration(self):
        """���� tokenizer ��ģ�͵ļ���"""
        text = "Hello, world! This is a test."
        
        # ����
        input_ids = self.tokenizer.encode(text, return_tensors="pt")
        
        # ��� token IDs �Ƿ��ڴʻ��Χ��
        self.assertTrue((input_ids >= 0).all())
        self.assertTrue((input_ids < self.args.vocab_size).all())
        
        # ģ��ǰ�򴫲�
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs["logits"]
        
        # ��������״
        expected_shape = (1, input_ids.shape[1], self.args.vocab_size)
        self.assertEqual(logits.shape, expected_shape)
        
        # ����
        decoded = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        self.assertIsInstance(decoded, str)


def run_tests():
    """�������в���"""
    # ���������׼�
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
    
    # ���в���
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
