"""
测试 tokenization corner cases - 刻意制造 decode-encode 不一致

方向 A: clean_up_tokenization_spaces=True (空格规整)
方向 B: special tokens 处理不一致
方向 C: 自然的 Unicode corner cases
"""

from transformers import AutoTokenizer

MODEL_PATH = "Qwen/Qwen2-VL-2B-Instruct"

print("加载 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

def test_cleanup_spaces():
    """方向 A: 测试 clean_up_tokenization_spaces=True"""
    print("\n" + "="*80)
    print("方向 A: clean_up_tokenization_spaces=True")
    print("="*80)
    
    test_cases = [
        ("多个空格", "hello     world"),
        ("多换行+空格", "hello \n\n world"),
        ("多个 tab", "hello\t\t\tworld"),
        ("混合空白", "hello  \t  \n  world"),
        ("尾部空格", "hello world   "),
        ("前导空格", "   hello world"),
        ("空格+换行组合", "hello  world\n\nhow  are  you"),
    ]
    
    found = []
    
    for desc, text in test_cases:
        print(f"\n测试: {desc}")
        print(f"  原始: {repr(text)}")
        
        # encode
        ids = tokenizer.encode(text, add_special_tokens=False)
        
        # decode with cleanup
        decoded = tokenizer.decode(ids, clean_up_tokenization_spaces=True)
        
        # re-encode
        ids2 = tokenizer.encode(decoded, add_special_tokens=False)
        
        print(f"  ids:  {ids}")
        print(f"  decoded: {repr(decoded)}")
        print(f"  ids2: {ids2}")
        
        if ids != ids2:
            print(f"  ❌❌❌ TOKEN MISMATCH! {len(ids)} → {len(ids2)}")
            found.append((desc, text))
        elif text != decoded:
            print(f"  ⚠️ TEXT CHANGED but tokens same")
            found.append((desc, text))
        else:
            print(f"  ✅ Consistent")
    
    return found

def test_special_tokens():
    """方向 B: special tokens 处理"""
    print("\n" + "="*80)
    print("方向 B: special tokens 处理")
    print("="*80)
    
    test_cases = [
        ("包含 special token", "<|im_start|>hello<|im_end|>"),
        ("VLM template", "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>test<|im_end|>"),
    ]
    
    found = []
    
    for desc, text in test_cases:
        print(f"\n测试: {desc}")
        print(f"  原始: {repr(text)}")
        
        # 场景 1: add=True 然后 skip=True
        print("\n  场景1: encode(add=True) → decode(skip=True) → encode(add=True)")
        ids1 = tokenizer.encode(text, add_special_tokens=True)
        decoded1 = tokenizer.decode(ids1, skip_special_tokens=True)
        ids1_2 = tokenizer.encode(decoded1, add_special_tokens=True)
        
        print(f"    ids1:  {ids1[:15]}... (len={len(ids1)})")
        print(f"    decoded: {repr(decoded1[:60])}")
        print(f"    ids1_2: {ids1_2[:15]}... (len={len(ids1_2)})")
        
        if ids1 != ids1_2:
            print(f"    ❌❌❌ TOKEN MISMATCH!")
            found.append((f"{desc} (场景1)", text))
        else:
            print(f"    ✅ Consistent")
        
        # 场景 2: add=False 然后混用 skip
        print("\n  场景2: encode(add=False) → decode(skip=True) → encode(add=False)")
        ids2 = tokenizer.encode(text, add_special_tokens=False)
        decoded2 = tokenizer.decode(ids2, skip_special_tokens=True)
        ids2_2 = tokenizer.encode(decoded2, add_special_tokens=False)
        
        print(f"    ids2:  {ids2[:15]}... (len={len(ids2)})")
        print(f"    decoded: {repr(decoded2[:60])}")
        print(f"    ids2_2: {ids2_2[:15]}... (len={len(ids2_2)})")
        
        if ids2 != ids2_2:
            print(f"    ❌❌❌ TOKEN MISMATCH!")
            found.append((f"{desc} (场景2)", text))
        else:
            print(f"    ✅ Consistent")
    
    return found

def test_unicode_corner_cases():
    """方向 C: Unicode corner cases"""
    print("\n" + "="*80)
    print("方向 C: Unicode corner cases")
    print("="*80)
    
    test_cases = [
        ("零宽空格", "hello\u200bworld"),
        ("NBSP", "hello\u00a0world"),
        ("零宽连字", "hello\u200dworld"),
        ("软连字符", "hello\u00adworld"),
        ("CRLF", "hello\r\nworld"),
        ("组合字符", "café"),
        ("Emoji序列", "👨‍👩‍👧‍👦"),
        ("全角空格", "hello　world"),
    ]
    
    found = []
    
    for desc, text in test_cases:
        print(f"\n测试: {desc}")
        print(f"  原始: {repr(text)}")
        
        ids = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(ids)
        ids2 = tokenizer.encode(decoded, add_special_tokens=False)
        
        print(f"  ids:  {ids}")
        print(f"  decoded: {repr(decoded)}")
        print(f"  ids2: {ids2}")
        
        if ids != ids2:
            print(f"  ❌❌❌ TOKEN MISMATCH!")
            found.append((desc, text))
        elif text != decoded:
            print(f"  ⚠️ TEXT CHANGED")
            found.append((desc, text))
        else:
            print(f"  ✅ Consistent")
    
    return found

def test_extreme_cases():
    """极端测试案例"""
    print("\n" + "="*80)
    print("极端测试案例")
    print("="*80)
    
    test_cases = [
        ("纯空格", "     "),
        ("纯 tab", "\t\t\t"),
        ("纯换行", "\n\n\n"),
        ("混合空白", "  \t\n  \t\n  "),
        ("空字符串", ""),
        ("只有 special token", "<|im_start|>"),
    ]
    
    found = []
    
    for desc, text in test_cases:
        print(f"\n测试: {desc}")
        print(f"  原始: {repr(text)}")
        
        # 默认参数
        ids = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(ids)
        ids2 = tokenizer.encode(decoded, add_special_tokens=False)
        
        print(f"  ids:  {ids}")
        print(f"  decoded: {repr(decoded)}")
        print(f"  ids2: {ids2}")
        
        if ids != ids2:
            print(f"  ❌❌❌ TOKEN MISMATCH!")
            found.append((desc, text))
        elif text != decoded:
            print(f"  ⚠️ TEXT CHANGED")
            found.append((desc, text))
        else:
            print(f"  ✅ Consistent")
        
        # 尝试 cleanup
        decoded_clean = tokenizer.decode(ids, clean_up_tokenization_spaces=True)
        ids2_clean = tokenizer.encode(decoded_clean, add_special_tokens=False)
        
        if ids != ids2_clean:
            print(f"  ❌ With cleanup: TOKEN MISMATCH!")
            if (desc + " (cleanup)", text) not in found:
                found.append((desc + " (cleanup)", text))
        elif decoded != decoded_clean:
            print(f"  ⚠️ With cleanup: TEXT CHANGED: {repr(decoded_clean)}")
            if (desc + " (cleanup)", text) not in found:
                found.append((desc + " (cleanup)", text))
    
    return found

if __name__ == "__main__":
    print("="*80)
    print("Qwen2-VL Tokenizer Corner Cases 测试")
    print("="*80)
    
    all_found = []
    
    # 测试所有方向
    all_found.extend(test_cleanup_spaces())
    all_found.extend(test_special_tokens())
    all_found.extend(test_unicode_corner_cases())
    all_found.extend(test_extreme_cases())
    
    # 总结
    print("\n" + "="*80)
    print("测试总结")
    print("="*80)
    
    if all_found:
        print(f"\n✅ 找到 {len(all_found)} 个可能导致不一致的案例:")
        print("")
        for i, (desc, text) in enumerate(all_found, 1):
            print(f"{i}. {desc}")
            print(f"   文本: {repr(text)}")
        
        print("\n" + "="*80)
        print("推荐：使用这些案例修改 reproduce_issue_12383_corner_case.py")
        print("="*80)
    else:
        print("\n⚠️ 没有找到明显的不一致案例")
        print("这说明 Qwen2-VL tokenizer 在默认设置下非常健壮")
        print("但这不意味着问题不存在 - 性能和设计问题依然需要解决！")
