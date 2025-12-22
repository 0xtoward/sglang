"""
复现 Issue #12383: VLM token-in-token-out 一致性问题

核心测试：
1. 第一轮用文本输入得到输出
2. 手动拼接 token_ids（模拟 RLHF）
3. 第二轮用 token_ids + image_data 输入
4. 观察服务器是否改变了 token 序列
"""

import requests
from transformers import AutoTokenizer

BASE_URL = "http://localhost:30000"
MODEL_PATH = "Qwen/Qwen2-VL-2B-Instruct"

# 不同的图片URL
image_url_1 = "https://picsum.photos/seed/test1/300/300"
image_url_2 = "https://picsum.photos/seed/test2/300/300" 
image_url_3 = "https://picsum.photos/seed/test3/300/300"

print("加载 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

def test_multi_turn_with_token_ids():
    """简单直接的测试"""
    
    # ========== 第 1 轮 ==========
    print("\n" + "="*80)
    print("第1轮：文本输入 + 图片1")
    print("="*80)
    
    text1 = "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>描述这张图<|im_end|>\n<|im_start|>assistant\n"
    
    response1 = requests.post(
        f"{BASE_URL}/generate",
        json={
            "text": text1,
            "image_data": [image_url_1],
            "sampling_params": {"temperature": 0, "max_new_tokens": 20}
        }
    ).json()
    
    output1 = response1["text"]
    print(f"输出: {output1}")
    
    # 手动拼接 token_ids（模拟 RLHF 从模型输出获取 token_ids 的场景）
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
    print(f"  Input non-image: len={len(non_img_input_1)}, tokens={non_img_input_1}")
    print(f"  Output non-image: len={len(non_img_output_1)}, tokens={non_img_output_1}")
    
    # ========== 第 2 轮 ==========
    print("\n" + "="*80)
    print("第2轮：token_ids 输入 + 图片1 + 图片2")
    print("="*80)
    
    # 拼接：第一轮输入 + 第一轮输出 + 第二轮用户输入
    text2_user = "<|im_end|>\n<|im_start|>user\n再来一张<|vision_start|><|image_pad|><|vision_end|>对比<|im_end|>\n<|im_start|>assistant\n"
    text2_ids = tokenizer.encode(text2_user)
    if text2_ids[0] == tokenizer.bos_token_id:
        text2_ids = text2_ids[1:]
    
    full_ids_2 = input_ids_1 + output_ids + text2_ids
    
    # 记录非图片 tokens
    non_img_input_2 = [t for t in full_ids_2 if t != 151655]
    print(f"\n[CLIENT] Turn 2 - Sending to server:")
    print(f"  Full input_ids: len={len(full_ids_2)}")
    print(f"  Non-image tokens: len={len(non_img_input_2)}")
    print(f"    First 30: {non_img_input_2[:30]}")
    print(f"    Last 30:  {non_img_input_2[-30:]}")
    
    response2 = requests.post(
        f"{BASE_URL}/generate",
        json={
            "input_ids": full_ids_2,
            "image_data": [image_url_1, image_url_2],  # 两张图
            "sampling_params": {"temperature": 0, "max_new_tokens": 25}
        }
    ).json()
    
    output2 = response2["text"]
    print(f"\n输出: {output2}")
    
    # 记录输出 tokens
    output_ids_2 = tokenizer.encode(output2)
    if output_ids_2[0] == tokenizer.bos_token_id:
        output_ids_2 = output_ids_2[1:]
    non_img_output_2 = [t for t in output_ids_2 if t != 151655]
    print(f"  Output non-image: len={len(non_img_output_2)}, tokens={non_img_output_2}")
    
    # ========== 第 3 轮 ==========
    print("\n" + "="*80)
    print("第3轮：token_ids 输入 + 三张图")
    print("="*80)
    
    text3_user = "<|im_end|>\n<|im_start|>user\n第三张<|vision_start|><|image_pad|><|vision_end|>总结<|im_end|>\n<|im_start|>assistant\n"
    text3_ids = tokenizer.encode(text3_user)
    if text3_ids[0] == tokenizer.bos_token_id:
        text3_ids = text3_ids[1:]
    
    full_ids_3 = full_ids_2 + output_ids_2 + text3_ids
    
    # 记录非图片 tokens
    non_img_input_3 = [t for t in full_ids_3 if t != 151655]
    print(f"\n[CLIENT] Turn 3 - Sending to server:")
    print(f"  Full input_ids: len={len(full_ids_3)}")
    print(f"  Non-image tokens: len={len(non_img_input_3)}")
    print(f"    First 30: {non_img_input_3[:30]}")
    print(f"    Last 30:  {non_img_input_3[-30:]}")
    
    response3 = requests.post(
        f"{BASE_URL}/generate",
        json={
            "input_ids": full_ids_3,
            "image_data": [image_url_1, image_url_2, image_url_3],  # 三张图
            "sampling_params": {"temperature": 0, "max_new_tokens": 30}
        }
    ).json()
    
    output3 = response3["text"]
    print(f"\n输出: {output3}")
    
    # 记录输出 tokens
    output_ids_3 = tokenizer.encode(output3)
    if output_ids_3[0] == tokenizer.bos_token_id:
        output_ids_3 = output_ids_3[1:]
    non_img_output_3 = [t for t in output_ids_3 if t != 151655]
    print(f"  Output non-image: len={len(non_img_output_3)}, tokens={non_img_output_3}")
    
    print("\n" + "="*80)
    print("✅ 测试完成！")
    print("="*80)
    print("\n🔍 查看服务端日志，寻找 [DEBUG-12383] 输出：")
    print("")
    print("关键检查点（第2轮和第3轮）：")
    print("")
    print("1. 对比客户端发送的非图片 tokens 和服务端 BEFORE 接收到的：")
    print("   - 应该完全一致")
    print("")
    print("2. 对比服务端 BEFORE 和 AFTER 的非图片 tokens：")
    print("   - 如果不一致，输出: ❌❌❌ NON-IMAGE TOKENS CHANGED! ❌❌❌")
    print("   - 这说明 decode-retokenize 破坏了 token 一致性")
    print("")
    print("3. 图片占位符 token (151655) 数量变化：")
    print("   - 这是预期行为（图片展开）")
    print("")
    print("="*80)


if __name__ == "__main__":
    test_multi_turn_with_token_ids()

