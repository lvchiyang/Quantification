#!/usr/bin/env python3
"""
������֤�ű�
��֤����ʵ�ֵĹ����Ƿ���������
"""

import sys
import os
import torch
import time
from typing import Dict, Any

# ��� src Ŀ¼��·��
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def print_section(title: str):
    """��ӡ�½ڱ���"""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)

def print_subsection(title: str):
    """��ӡ���½ڱ���"""
    print(f"\n--- {title} ---")

def test_imports() -> bool:
    """��������ģ�鵼��"""
    print_subsection("����ģ�鵼��")
    
    try:
        # ����ģ��
        from config import ModelArgs, ModelConfigs
        from utils import RMSNorm, precompute_freqs_cis, apply_rotary_emb
        from attention import MLA, MultiHeadAttention
        from feedforward import SwiGLU, TransformerBlock
        from transformer import Transformer
        from trainer import Trainer
        from model_utils import validate_model_architecture
        
        print("? ���к���ģ�鵼��ɹ�")
        return True
    except Exception as e:
        print(f"? ģ�鵼��ʧ��: {e}")
        return False

def test_model_configurations() -> bool:
    """���Բ�ͬģ������"""
    print_subsection("����ģ������")
    
    try:
        from config import ModelConfigs
        from transformer import Transformer
        
        configs = {
            "tiny": ModelConfigs.tiny(),
            "small": ModelConfigs.small(),
            "base": ModelConfigs.base()
        }
        
        for name, args in configs.items():
            model = Transformer(args)
            param_count = sum(p.numel() for p in model.parameters())
            print(f"? {name.upper()} ģ��: {param_count:,} ����")
        
        return True
    except Exception as e:
        print(f"? ģ�����ò���ʧ��: {e}")
        return False

def test_forward_pass() -> bool:
    """����ǰ�򴫲�"""
    print_subsection("����ǰ�򴫲�")
    
    try:
        from config import ModelConfigs
        from transformer import Transformer
        
        args = ModelConfigs.tiny()
        model = Transformer(args)
        
        # ���Բ�ͬ���δ�С�����г���
        test_cases = [
            (1, 16),   # ��������������
            (2, 32),   # С���Σ��е�����
            (4, 64),   # �е����Σ�������
        ]
        
        for batch_size, seq_len in test_cases:
            input_ids = torch.randint(0, args.vocab_size, (batch_size, seq_len))
            
            with torch.no_grad():
                outputs = model(input_ids)
                logits = outputs["logits"]
            
            expected_shape = (batch_size, seq_len, args.vocab_size)
            assert logits.shape == expected_shape, f"��״��ƥ��: {logits.shape} vs {expected_shape}"
            assert torch.isfinite(logits).all(), "�������������ֵ"
            
            print(f"? ���δ�С {batch_size}, ���г��� {seq_len}: {logits.shape}")
        
        return True
    except Exception as e:
        print(f"? ǰ�򴫲�����ʧ��: {e}")
        return False

def test_loss_computation() -> bool:
    """������ʧ����"""
    print_subsection("������ʧ����")
    
    try:
        from config import ModelConfigs
        from transformer import Transformer
        
        args = ModelConfigs.tiny()
        model = Transformer(args)
        
        input_ids = torch.randint(0, args.vocab_size, (2, 32))
        
        # ������ʧ����
        outputs = model(input_ids, labels=input_ids)
        loss = outputs["loss"]
        
        assert loss is not None, "��ʧΪ None"
        assert torch.isfinite(loss), "��ʧ��������ֵ"
        assert loss.item() > 0, "��ʧӦ��Ϊ����"
        
        print(f"? ��ʧ����ɹ�: {loss.item():.4f}")
        
        # �����ݶȼ���
        loss.backward()
        
        grad_count = 0
        for param in model.parameters():
            if param.requires_grad and param.grad is not None:
                grad_count += 1
        
        print(f"? �ݶȼ���ɹ�: {grad_count} ���������ݶ�")
        
        return True
    except Exception as e:
        print(f"? ��ʧ�������ʧ��: {e}")
        return False

def test_text_generation() -> bool:
    """�����ı�����"""
    print_subsection("�����ı�����")
    
    try:
        from config import ModelConfigs
        from transformer import Transformer
        
        args = ModelConfigs.tiny()
        model = Transformer(args)
        model.eval()
        
        # ���Բ�ͬ���ɲ���
        test_cases = [
            {"do_sample": False, "max_new_tokens": 10},  # ̰�Ľ���
            {"do_sample": True, "temperature": 1.0, "max_new_tokens": 10},  # ����
            {"do_sample": True, "top_k": 10, "max_new_tokens": 10},  # Top-K
            {"do_sample": True, "top_p": 0.9, "max_new_tokens": 10},  # Top-P
        ]
        
        for i, gen_kwargs in enumerate(test_cases):
            input_ids = torch.randint(0, args.vocab_size, (1, 5))
            
            with torch.no_grad():
                generated = model.generate(input_ids=input_ids, **gen_kwargs)
            
            assert generated.shape[0] == 1, "���δ�С����ȷ"
            assert generated.shape[1] > input_ids.shape[1], "û�������µ� token"
            assert (generated >= 0).all(), "���ɵ� token ID Ϊ����"
            assert (generated < args.vocab_size).all(), "���ɵ� token ID �����ʻ��"
            
            print(f"? ���ɲ��� {i+1}: {input_ids.shape[1]} -> {generated.shape[1]} tokens")
        
        return True
    except Exception as e:
        print(f"? �ı����ɲ���ʧ��: {e}")
        return False

def test_model_validation() -> bool:
    """����ģ����֤"""
    print_subsection("����ģ����֤")
    
    try:
        from config import ModelConfigs
        from transformer import Transformer
        from model_utils import validate_model_architecture
        
        args = ModelConfigs.tiny()
        model = Transformer(args)
        
        validation_results = validate_model_architecture(model, args)
        
        if validation_results["passed"]:
            print("? ģ�ͼܹ���֤ͨ��")
            print(f"  �ܲ�����: {validation_results['info']['total_parameters']:,}")
            print(f"  ������ʧ: {validation_results['info']['sample_loss']:.4f}")
        else:
            print("? ģ�ͼܹ���֤ʧ��")
            for error in validation_results["errors"]:
                print(f"  ����: {error}")
            return False
        
        return True
    except Exception as e:
        print(f"? ģ����֤����ʧ��: {e}")
        return False

def test_component_functionality() -> bool:
    """���Ը����������"""
    print_subsection("�����������")
    
    try:
        from config import ModelConfigs
        from utils import RMSNorm, precompute_freqs_cis
        from attention import MLA
        from feedforward import SwiGLU
        
        args = ModelConfigs.tiny()
        batch_size, seq_len = 2, 32
        
        # ���� RMSNorm
        norm = RMSNorm(args.d_model)
        x = torch.randn(batch_size, seq_len, args.d_model)
        normed = norm(x)
        assert normed.shape == x.shape, "RMSNorm �����״����ȷ"
        print("? RMSNorm ����ͨ��")
        
        # ���� RoPE
        freqs_cis = precompute_freqs_cis(args.qk_rope_head_dim, seq_len)
        assert freqs_cis.shape == (seq_len, args.qk_rope_head_dim // 2), "RoPE Ƶ����״����ȷ"
        print("? RoPE ����ͨ��")
        
        # ���� MLA
        mla = MLA(args)
        attn_out = mla(x, freqs_cis)
        assert attn_out.shape == x.shape, "MLA �����״����ȷ"
        print("? MLA ����ͨ��")
        
        # ���� SwiGLU
        ffn = SwiGLU(args)
        ffn_out = ffn(x)
        assert ffn_out.shape == x.shape, "SwiGLU �����״����ȷ"
        print("? SwiGLU ����ͨ��")
        
        return True
    except Exception as e:
        print(f"? ������ܲ���ʧ��: {e}")
        return False

def test_performance() -> bool:
    """��������"""
    print_subsection("��������")
    
    try:
        from config import ModelConfigs
        from transformer import Transformer
        
        args = ModelConfigs.tiny()
        model = Transformer(args)
        model.eval()
        
        # ����ǰ�򴫲��ٶ�
        input_ids = torch.randint(0, args.vocab_size, (4, 128))
        
        # Ԥ��
        with torch.no_grad():
            for _ in range(3):
                model(input_ids)
        
        # ����ʱ��
        start_time = time.time()
        with torch.no_grad():
            for _ in range(10):
                outputs = model(input_ids)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 10
        tokens_per_second = (4 * 128) / avg_time
        
        print(f"? ǰ�򴫲�����: {avg_time*1000:.2f}ms, {tokens_per_second:.0f} tokens/s")
        
        # ���������ٶ�
        input_ids = torch.randint(0, args.vocab_size, (1, 10))
        
        start_time = time.time()
        with torch.no_grad():
            generated = model.generate(input_ids, max_new_tokens=20, do_sample=False)
        end_time = time.time()
        
        gen_time = end_time - start_time
        gen_tokens_per_second = 20 / gen_time
        
        print(f"? ��������: {gen_time*1000:.2f}ms, {gen_tokens_per_second:.1f} tokens/s")
        
        return True
    except Exception as e:
        print(f"? ���ܲ���ʧ��: {e}")
        return False

def main():
    """������"""
    print_section("Decoder-only Transformer with MLA - ������֤")
    
    print("�����֤�ű�����������ʵ�ֵĹ���...")
    
    # �����б�
    tests = [
        ("ģ�鵼��", test_imports),
        ("ģ������", test_model_configurations),
        ("ǰ�򴫲�", test_forward_pass),
        ("��ʧ����", test_loss_computation),
        ("�ı�����", test_text_generation),
        ("ģ����֤", test_model_validation),
        ("�������", test_component_functionality),
        ("���ܲ���", test_performance),
    ]
    
    # ���в���
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print_section(f"����: {test_name}")
        try:
            if test_func():
                passed += 1
                print(f"\n? {test_name} ����ͨ��")
            else:
                print(f"\n? {test_name} ����ʧ��")
        except Exception as e:
            print(f"\n? {test_name} �����쳣: {e}")
    
    # �ܽ�
    print_section("��֤���")
    print(f"���Խ��: {passed}/{total} ͨ��")
    
    if passed == total:
        print("\n? ���в���ͨ����")
        print("Decoder-only Transformer with MLA ʵ����ȫ��ȷ��")
        print("\n? ��Ŀ�ص�:")
        print("  ? Pre-RMSNorm Ԥ��һ���ṹ")
        print("  ? MLA ��ͷǱ��ע��������")
        print("  ? RoPE ��תλ�ñ���")
        print("  ? SwiGLU �ſ����Ե�Ԫ")
        print("  ? ������ѵ������������")
        print("\n? ��һ��:")
        print("  1. ���� python examples/demo.py �鿴������ʾ")
        print("  2. ���� python examples/train.py --model_size tiny ��ʼѵ��")
        print("  3. �鿴 PROJECT_SUMMARY.md �˽���ϸ��Ϣ")
    else:
        print(f"\n? {total - passed} ������ʧ��")
        print("����ʵ�ֲ��޸�����")
    
    print_section("��֤���")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
