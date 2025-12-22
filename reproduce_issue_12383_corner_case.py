"""
复现 Issue #12383 - 使用 corner case 文本触发 token 不一致

策略：
1. 只使用一张图片（所有轮次都复用）
2. 在对话文本中插入 tokenization corner cases
3. 观察服务器是否改变了非图片 token
"""

import requests
from transformers import AutoTokenizer

BASE_URL = "http://localhost:30000"
MODEL_PATH = "Qwen/Qwen2-VL-2B-Instruct"
image_url = "https://picsum.photos/seed/cornercase/300/300"

print("加载 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

# 已知的 tokenization corner cases
CORNER_CASES = {
    "连续空格": "有很多     空格",
    "NBSP": "non-breaking\u00a0space",
    "零宽空格": "zero\u200bwidth\u200bspace",
    "CRLF": "Windows\r\n换行",
    "全角空格": "中文　空格",
    "混合": "测试\u00a0\u200b混合\r\n场景",
}

def test_multi_turn_single_image():
    """单图多轮对话，文本包含 corner cases"""
    
    # ========== 第 1 轮 ==========
    print("\n" + "="*80)
    print("第1轮：文本输入 + 图片")
    print("="*80)
    
    text1 = "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>描述这张图<|im_end|>\n<|im_start|>assistant\n"
    
    response1 = requests.post(
        f"{BASE_URL}/generate",
        json={
            "text": text1,
            "image_data": [image_url],
            "sampling_params": {"temperature": 0, "max_new_tokens": 15}
        }
    ).json()
    
    output1 = response1["text"]
    print(f"输出: {output1}")
    
    # 准备 token_ids
    output_ids = tokenizer.encode(output1)
    if output_ids[0] == tokenizer.bos_token_id:
        output_ids = output_ids[1:]
    
    input_ids_1 = tokenizer.encode(text1)
    if input_ids_1[0] == tokenizer.bos_token_id:
        input_ids_1 = input_ids_1[1:]
    
    # 记录非图片 tokens
    non_img_input_1 = [t for t in input_ids_1 if t != 151655]
    non_img_output_1 = [t for t in output_ids if t != 151655]
    print(f"\n[CLIENT] Turn 1 tokens:")
    print(f"  Input non-image: len={len(non_img_input_1)}")
    print(f"  Output non-image: len={len(non_img_output_1)}")
    
    # ========== 第 2 轮：测试各个 corner case ==========
    for case_name, case_text in CORNER_CASES.items():
        print("\n" + "="*80)
        print(f"第2轮：Corner Case = {case_name}")
        print(f"  文本: {repr(case_text)}")
        print("="*80)
        
        # 构造包含 corner case 的用户输入
        text2_user = f"<|im_end|>\n<|im_start|>user\n{case_text}<|im_end|>\n<|im_start|>assistant\n"
        text2_ids = tokenizer.encode(text2_user)
        if text2_ids[0] == tokenizer.bos_token_id:
            text2_ids = text2_ids[1:]
        
        # 拼接完整历史
        full_ids_2 = input_ids_1 + output_ids + text2_ids
        
        # 记录客户端发送的非图片 tokens
        non_img_input_2 = [t for t in full_ids_2 if t != 151655]
        print(f"\n[CLIENT] Turn 2 - Sending to server:")
        print(f"  Full input_ids: len={len(full_ids_2)}")
        print(f"  Non-image tokens: len={len(non_img_input_2)}")
        print(f"    First 20: {non_img_input_2[:20]}")
        print(f"    Last 20:  {non_img_input_2[-20:]}")
        
        # 发送请求（复用同一张图片）
        response2 = requests.post(
            f"{BASE_URL}/generate",
            json={
                "input_ids": full_ids_2,
                "image_data": [image_url],  # 复用第1轮的图片
                "sampling_params": {"temperature": 0, "max_new_tokens": 15}
            }
        ).json()
        
        output2 = response2["text"]
        print(f"\n输出: {output2}")
        
        # 记录输出 tokens
        output_ids_2 = tokenizer.encode(output2)
        if output_ids_2[0] == tokenizer.bos_token_id:
            output_ids_2 = output_ids_2[1:]
        non_img_output_2 = [t for t in output_ids_2 if t != 151655]
        print(f"  Output non-image: len={len(non_img_output_2)}")
        
        print(f"\n⚠️ 查看服务器日志 [DEBUG-12383]，检查此 corner case 是否触发 TOKEN MISMATCH")
        print("-" * 80)
    
    # ========== 第 3 轮：组合多个 corner cases ==========
    print("\n" + "="*80)
    print("第3轮：组合多个 corner cases")
    print("="*80)
    
    # 使用最后一个 corner case 的输出继续
    combined_text = "测试组合：" + "、".join(CORNER_CASES.values())
    text3_user = f"<|im_end|>\n<|im_start|>user\n{combined_text}<|im_end|>\n<|im_start|>assistant\n"
    text3_ids = tokenizer.encode(text3_user)
    if text3_ids[0] == tokenizer.bos_token_id:
        text3_ids = text3_ids[1:]
    
    # 拼接（使用第1轮的历史）
    full_ids_3 = input_ids_1 + output_ids + text3_ids
    
    non_img_input_3 = [t for t in full_ids_3 if t != 151655]
    print(f"\n[CLIENT] Turn 3 - Sending to server:")
    print(f"  Full input_ids: len={len(full_ids_3)}")
    print(f"  Non-image tokens: len={len(non_img_input_3)}")
    print(f"  Combined text: {repr(combined_text)}")
    
    response3 = requests.post(
        f"{BASE_URL}/generate",
        json={
            "input_ids": full_ids_3,
            "image_data": [image_url],  # 继续复用同一张图片
            "sampling_params": {"temperature": 0, "max_new_tokens": 20}
        }
    ).json()
    
    output3 = response3["text"]
    print(f"\n输出: {output3}")
    
    print("\n" + "="*80)
    print("✅ 测试完成！")
    print("="*80)
    print("\n📊 测试了以下 corner cases:")
    for name in CORNER_CASES.keys():
        print(f"  - {name}")
    print("\n🔍 检查服务器日志中的 [DEBUG-12383] 输出")
    print("   特别关注是否出现: ❌❌❌ NON-IMAGE TOKENS CHANGED!")
    print("="*80)

if __name__ == "__main__":
    test_multi_turn_single_image()


