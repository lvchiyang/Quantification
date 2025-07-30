# -*- coding: utf-8 -*-
"""
测试README.md更新是否成功
"""

import os


def test_readme_content():
    """测试README.md内容是否正确更新"""
    print("🧪 测试README.md更新")
    print("=" * 50)
    
    readme_path = "README.md"
    
    if not os.path.exists(readme_path):
        print("❌ README.md文件不存在")
        return False
    
    with open(readme_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查关键内容是否存在
    checks = [
        ("金融时序预测系统", "项目标题"),
        ("Multi-Head Latent Attention", "MLA技术"),
        ("RoPE 位置编码", "RoPE技术"),
        ("SwiGLU 前馈网络", "SwiGLU技术"),
        ("20维金融特征", "特征维度"),
        ("未来10个时间点预测", "预测时间点"),
        ("金融专用损失函数", "损失函数"),
        ("FinancialMultiLoss", "损失函数类"),
        ("PricePredictionConfigs", "配置类"),
        ("doc/transformer.md", "文档链接"),
        ("doc/financial_losses.md", "损失函数文档"),
        ("doc/config.md", "配置文档"),
        ("test/test_financial_losses.py", "测试文件"),
    ]
    
    passed = 0
    total = len(checks)
    
    for keyword, description in checks:
        if keyword in content:
            print(f"  ✅ {description}: 找到 '{keyword}'")
            passed += 1
        else:
            print(f"  ❌ {description}: 未找到 '{keyword}'")
    
    print(f"\n检查结果: {passed}/{total} 项通过")
    
    # 检查文档结构
    print(f"\n📋 文档结构检查:")
    
    sections = [
        "# 🚀 金融时序预测系统",
        "## ✨ 核心特性", 
        "## 🏗️ 项目结构",
        "## 🚀 快速开始",
        "## 📊 技术架构",
        "## 📈 数据格式",
        "## 🎯 金融专用损失函数",
        "## 🎯 使用示例",
        "## 🔧 高级配置",
        "## 📚 完整文档",
        "## 🧪 测试和验证",
        "## 🤝 贡献指南",
        "## 📄 许可证",
        "## 🙏 致谢",
    ]
    
    section_passed = 0
    for section in sections:
        if section in content:
            print(f"  ✅ {section}")
            section_passed += 1
        else:
            print(f"  ❌ {section}")
    
    print(f"\n章节检查: {section_passed}/{len(sections)} 个章节存在")
    
    # 统计信息
    lines = content.split('\n')
    print(f"\n📊 文档统计:")
    print(f"  总行数: {len(lines)}")
    print(f"  总字符数: {len(content)}")
    print(f"  代码块数: {content.count('```')//2}")
    
    # 检查是否有编码问题
    encoding_issues = 0
    for i, line in enumerate(lines, 1):
        if '�' in line:
            print(f"  ⚠️  第{i}行有编码问题: {line.strip()}")
            encoding_issues += 1
    
    if encoding_issues == 0:
        print(f"  ✅ 无编码问题")
    else:
        print(f"  ❌ 发现 {encoding_issues} 个编码问题")
    
    # 总体评估
    overall_score = (passed / total + section_passed / len(sections)) / 2
    
    print(f"\n{'='*50}")
    print(f"总体评估: {overall_score:.1%}")
    
    if overall_score >= 0.9:
        print("🎉 README.md更新成功！")
        return True
    elif overall_score >= 0.7:
        print("⚠️  README.md基本更新成功，但有部分内容需要完善")
        return True
    else:
        print("❌ README.md更新不完整，需要进一步修改")
        return False


def check_doc_files():
    """检查doc目录下的文档文件"""
    print(f"\n📚 检查文档文件")
    print("=" * 50)
    
    doc_files = [
        "doc/architecture.md",
        "doc/transformer.md", 
        "doc/feedforward.md",
        "doc/embedding.md",
        "doc/financial_losses.md",
        "doc/sequences.md",
        "doc/config.md",
        "doc/data.md",
        "doc/training.md",
        "doc/troubleshooting.md"
    ]
    
    existing_files = []
    missing_files = []
    
    for doc_file in doc_files:
        if os.path.exists(doc_file):
            existing_files.append(doc_file)
            print(f"  ✅ {doc_file}")
        else:
            missing_files.append(doc_file)
            print(f"  ❌ {doc_file}")
    
    print(f"\n文档文件统计:")
    print(f"  存在: {len(existing_files)}/{len(doc_files)}")
    print(f"  缺失: {len(missing_files)}")
    
    if missing_files:
        print(f"\n缺失的文档:")
        for file in missing_files:
            print(f"    - {file}")
    
    return len(missing_files) == 0


if __name__ == "__main__":
    print("🎯 README.md更新验证")
    print("=" * 60)
    
    # 测试README内容
    readme_ok = test_readme_content()
    
    # 检查文档文件
    docs_ok = check_doc_files()
    
    print(f"\n{'='*60}")
    print(f"验证结果:")
    print(f"  README.md: {'✅ 通过' if readme_ok else '❌ 失败'}")
    print(f"  文档文件: {'✅ 完整' if docs_ok else '⚠️  部分缺失'}")
    
    if readme_ok and docs_ok:
        print("\n🎉 所有检查通过！项目文档已完整更新")
    elif readme_ok:
        print("\n⚠️  README.md更新成功，但部分文档文件缺失")
    else:
        print("\n❌ 需要进一步完善README.md和文档")
