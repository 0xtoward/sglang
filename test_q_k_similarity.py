#!/usr/bin/env python3
"""
测试用 K[last] 代替 Q[last] 的合理性

实验设计：
1. 加载一个真实的 Transformer 模型
2. 对一个句子进行 forward
3. 在某一层提取 Q 和 K
4. 计算：
   - 真实 attention: softmax(Q[last] @ K.T)
   - 近似 attention: softmax(K[last] @ K.T)
5. 比较两者的相关性
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from scipy.stats import spearmanr, pearsonr

def test_q_k_approximation(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", layer_idx=10):
    """测试 Q 和 K 的近似关系"""
    
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16,
        device_map="cuda"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 测试文本
    text = "The quick brown fox jumps over the lazy dog. This is a test sentence for attention analysis."
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    
    # Forward pass with output_hidden_states
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, use_cache=True)
    
    # 获取指定层的 attention 模块
    layer = model.model.layers[layer_idx]
    attn_module = layer.self_attn
    
    # 获取 hidden states
    hidden_states = outputs.hidden_states[layer_idx]  # 在 layer_idx 之前的输出
    print(f"Hidden states shape: {hidden_states.shape}")
    
    # 计算 Q 和 K
    q = attn_module.q_proj(hidden_states)
    k = attn_module.k_proj(hidden_states)
    
    # Reshape
    batch_size, seq_len, _ = hidden_states.shape
    num_heads = attn_module.config.num_attention_heads
    head_dim = attn_module.head_dim
    
    q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    k = k.view(batch_size, seq_len, attn_module.config.num_key_value_heads, head_dim).transpose(1, 2)
    
    print(f"Q shape: {q.shape}, K shape: {k.shape}")
    
    # 提取最后一个 token
    q_last = q[:, :, -1:, :]  # [1, num_heads, 1, head_dim]
    k_last = k[:, :, -1:, :]  # [1, num_kv_heads, 1, head_dim]
    
    # 计算两种 attention scores
    # 1. 真实: Q[last] @ K.T
    scores_real = torch.matmul(q_last, k.transpose(2, 3)) / (head_dim ** 0.5)
    scores_real = scores_real.squeeze().cpu().numpy()  # [num_heads, seq_len]
    
    # 2. 近似: K[last] @ K.T
    scores_approx = torch.matmul(k_last, k.transpose(2, 3)) / (head_dim ** 0.5)
    scores_approx = scores_approx.squeeze().cpu().numpy()  # [num_kv_heads, seq_len]
    
    print(f"\nReal scores shape: {scores_real.shape}")
    print(f"Approx scores shape: {scores_approx.shape}")
    
    # 计算相关性（对每个 head）
    pearson_corrs = []
    spearman_corrs = []
    
    # 如果是 GQA，需要扩展 k_last
    num_groups = num_heads // k.shape[1]
    if num_groups > 1:
        scores_approx_expanded = np.repeat(scores_approx, num_groups, axis=0)
    else:
        scores_approx_expanded = scores_approx
    
    for head_idx in range(min(num_heads, 8)):  # 只测试前 8 个 heads
        real = scores_real[head_idx]
        approx = scores_approx_expanded[head_idx]
        
        pearson_r, _ = pearsonr(real, approx)
        spearman_r, _ = spearmanr(real, approx)
        
        pearson_corrs.append(pearson_r)
        spearman_corrs.append(spearman_r)
        
        print(f"Head {head_idx}: Pearson={pearson_r:.3f}, Spearman={spearman_r:.3f}")
    
    # 总体统计
    print(f"\n=== Summary ===")
    print(f"Average Pearson correlation: {np.mean(pearson_corrs):.3f} ± {np.std(pearson_corrs):.3f}")
    print(f"Average Spearman correlation: {np.mean(spearman_corrs):.3f} ± {np.std(spearman_corrs):.3f}")
    
    # 计算 Top-K 一致性（对于压缩任务更重要）
    k_values = [5, 10, 20]
    print(f"\n=== Top-K Overlap (for compression) ===")
    for k_val in k_values:
        overlaps = []
        for head_idx in range(min(num_heads, 8)):
            real = scores_real[head_idx]
            approx = scores_approx_expanded[head_idx]
            
            # 获取 top-k indices
            topk_real = np.argsort(real)[-k_val:]
            topk_approx = np.argsort(approx)[-k_val:]
            
            # 计算重叠
            overlap = len(set(topk_real) & set(topk_approx)) / k_val
            overlaps.append(overlap)
        
        print(f"Top-{k_val} overlap: {np.mean(overlaps):.1%} ± {np.std(overlaps):.1%}")
    
    return {
        'pearson_mean': np.mean(pearson_corrs),
        'spearman_mean': np.mean(spearman_corrs),
        'scores_real': scores_real,
        'scores_approx': scores_approx_expanded
    }

if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/home/l1q/WSL/sglang/python')
    
    try:
        results = test_q_k_approximation()
        print("\n✅ Test completed successfully!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

