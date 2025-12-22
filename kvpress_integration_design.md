# KVPress for SGLang：集成设计文档

## 项目概述

将 NVIDIA 开源的 KVPress 库集成到 SGLang 推理框架，实现**注意力机制无关**的 KV cache 压缩。

---

## 技术背景

### 现有方案的局限

**1. 稀疏注意力**（NSA, MInference, DSA）
- 仅减少计算量（O(N²) → O(N log N)）
- **不压缩显存**：仍存储所有 token 的 KV
- 示例：
  - NSA (DeepSeek V3)：动态选择 topk token 参与 attention
  - MInference：垂直+斜线稀疏模式
  - DSA：通道稀疏 + token 稀疏

**2. 架构级优化**（DeepSeek V3）
- 需要修改模型结构（MLA、NSA Indexer）
- **不通用**，无法应用于已有模型

### KVPress 的优势

- ✅ **注意力无关**：适用于任意 Transformer 模型
- ✅ **真正压缩显存**：物理删除 token，释放 KV cache 空间
- ✅ **插件化设计**：支持 10+ 种启发式算法
- ✅ **可配置**：compression_ratio 灵活调节（30%-70%）

**性能数据（来自 KVPress Leaderboard）：**
| Method | Compression | RULER 4K Score | 显存节约 |
|--------|-------------|----------------|---------|
| Baseline | 0% | 95% | 0% |
| KnormPress | 50% | 90% | ~44% |
| SnapKVPress | 50% | 92% | ~44% |
| CriticalKVPress | 50% | 94% | ~44% |

---

## 技术挑战

### 1. KVPress 工作流程理解

#### 核心机制
```python
# KVPress 的三步流程（逐层执行）
for layer in model.layers:
    # Step 1: 计算重要性分数
    scores = press.score(keys, values, hidden_states)
    # scores.shape = (batch, heads, seq_len)
    
    # Step 2: TopK 选择
    n_kept = int(seq_len * (1 - compression_ratio))
    indices = scores.topk(n_kept, dim=-1).indices
    
    # Step 3: Gather 压缩
    keys = keys.gather(2, indices).contiguous()
    values = values.gather(2, indices).contiguous()
    cache[layer].keys = keys  # 更新 cache
```

#### 关键特性
- **逐层压缩**：每层 forward 后立即触发 `forward_hook`
- **Token-wise**：大部分方法是删除整个 token（所有 head 一致）
- **Head-wise**（少数）：如 `AdaKVPress`，每个 head 保留不同 token
- **HuggingFace 依赖**：基于 `DynamicCache` 和 `register_forward_hook`

#### 支持的压缩方法（Scorer-based）

| Method | Score 定义 | 特点 |
|--------|-----------|------|
| **KnormPress** | `-‖key‖₂` | 最快，通用 |
| **SnapKVPress** | `mean(attention[-64:, :])` | 需要 attention weights |
| **ExpectedAttentionPress** | `E[softmax(q·k)]` | 预测未来 attention |
| **ObservedAttentionPress** | `mean(attention_all)` | Prefill 阶段观察到的 attention |
| **CriticalKVPress** | `L1(Wo @ values)` | 两阶段精细选择，最优 |

---

### 2. SGLang 显存管理机制

#### 两级内存映射

```
┌─────────────────────────────────────────────────────────┐
│  GPU 端（Device）                                        │
│  ┌────────────────────────────────────────────────┐    │
│  │ 第一级：ReqToTokenPool (逻辑映射)               │    │
│  │ req_to_token[req_idx, token_pos] → slot_id    │    │
│  │ 形状：(max_num_reqs, max_context_len)          │    │
│  │ 作用：将请求内的 token 位置映射到物理槽位       │    │
│  └────────────────────────────────────────────────┘    │
│                        ↓                                 │
│  ┌────────────────────────────────────────────────┐    │
│  │ 第二级：KVCache (物理存储)                      │    │
│  │ k_buffer[layer_id].shape = (size, heads, dim)  │    │
│  │ v_buffer[layer_id].shape = (size, heads, dim)  │    │
│  │ 索引：k_buffer[layer][slot_id] → K 向量        │    │
│  │ 作用：实际存储 K/V 数据                         │    │
│  └────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

**关键点：**
- `slot_id` 是全局物理索引，垂直贯穿所有层
- 同一个 `slot_id` 在不同层存储不同的 K/V 向量
- MHA: `k_buffer[layer][slot_id, :, :]` 是 `(heads, dim)` 的向量
- MLA: `k_buffer[layer].shape = (size, 1, kv_lora_rank + qk_rope_head_dim)`

#### Allocator 机制

```python
class TokenToKVPoolAllocator:
    def alloc(self, num_tokens: int) -> torch.Tensor:
        """分配连续的 slot_ids"""
        if len(self.free_pages) < num_tokens:
            if self.need_sort:
                self.merge_and_sort_free()  # 碎片整理
        return self.free_pages[:num_tokens]
    
    def free(self, indices: torch.Tensor):
        """释放 slot_ids，标记为可重用"""
        if self.need_sort:
            self.release_pages = torch.cat([self.release_pages, indices])
        else:
            self.free_pages = torch.cat([self.free_pages, indices])
```

**分配模式：**
- **Token-level** (`page_size=1`)：逐 token 分配，灵活但可能碎片化
- **Paged** (`page_size>1`)：按页分配，连续但有内部碎片

#### RadixCache（前缀共享）

```python
# 多个请求共享相同前缀的 KV cache
req1 = "Context A + Question 1"
req2 = "Context A + Question 2"
# "Context A" 的 KV cache 被共享
```

**对 KVPress 的影响：**
- ❌ 压缩后 KV 长度不一致，破坏前缀匹配
- 🔧 解决方案：Session 模式下禁用 prefix sharing，或只对 context 压缩一次

#### HiCache（多级缓存）

```
GPU KV Cache (热数据)
    ↓ evict
Host Memory (温数据)
    ↓ backup
External Storage (冷数据, SSD/网络)
```

**KVPress 的适用场景：**
- 主要在 GPU 端压缩
- Host/External 端的 transfer 可能需要额外适配

---

### 3. 集成关键点

#### 挑战 1：SGLang 无 `forward_hook`

**原因：** SGLang 使用 `torch.compile` 和 CUDA Graph，不支持动态 hook

**解决方案：** 在 `set_kv_buffer` 前插入压缩逻辑

```python
# 文件：python/sglang/srt/layers/attention/flashattention_backend.py
def forward(...):
    # ... 计算 attention
    
    if k is not None and save_kv_cache:
        cache_loc = forward_batch.out_cache_loc
        
        # ✅ 插入点：在写入 cache 前压缩
        if enable_kvpress and forward_batch.forward_mode.is_extend():
            k, v, kept_indices = kvpress_compress(
                layer=layer,
                keys=k,
                values=v,
                hidden_states=hidden_states,
                layer_idx=layer.layer_idx,
                compression_ratio=forward_batch.kvpress_ratio
            )
            # 更新 cache_loc，只写入保留的 token
            cache_loc = cache_loc[kept_indices]
            
            # 🔑 关键：释放被剪枝的 slot
            pruned_indices = get_pruned_indices(cache_loc, kept_indices)
            forward_batch.token_to_kv_pool_allocator.free(pruned_indices)
        
        # 写入压缩后的 KV
        forward_batch.token_to_kv_pool.set_kv_buffer(
            layer, cache_loc, k, v
        )
```

#### 挑战 2：内存真正释放

**问题：** 只压缩 tensor 不会释放显存，必须调用 `allocator.free()`

**流程：**
```python
# 1. 压缩前：10 个 token，占用 slot [5, 8, 12, 15, 20, 23, 27, 30, 33, 36]
original_loc = forward_batch.out_cache_loc  # [5, 8, 12, ...]

# 2. 压缩：保留 6 个 token（compression_ratio=0.4）
kept_indices = topk_indices  # [0, 2, 4, 6, 8, 9]
kept_loc = original_loc[kept_indices]  # [5, 12, 20, 27, 33, 36]

# 3. 计算被剪枝的 slot
pruned_loc = original_loc[~kept_indices]  # [8, 15, 23, 30]

# 4. ✅ 释放显存
allocator.free(pruned_loc)

# 5. 更新 req_to_token 映射
# 需要压缩 req_to_token，移除被剪枝的位置
req_to_token[req_idx, :new_len] = kept_loc
```

#### 挑战 3：RadixCache 兼容性

**方案 A：禁用前缀共享（保守）**
```python
if req.kvpress_enabled:
    # 不进行前缀匹配，独立分配 KV cache
    req.prefix_indices = torch.empty(0)
```

**方案 B：Session 模式（推荐）**
```python
# 第一次请求：压缩 context
req1 = generate(
    text=long_context + question1,
    session_params={"id": "session_1"},
    kvpress_params={"method": "knorm", "compression_ratio": 0.5}
)
# 压缩后的 context 存入 RadixCache

# 后续请求：复用压缩后的 context
req2 = generate(
    text=question2,
    session_params={"id": "session_1", "drop_previous_output": True},
)
```

#### 挑战 4：Per-request 配置

**API 设计：**
```python
@dataclass
class GenerateReqInput(BaseReq):
    # 现有字段...
    text: Optional[Union[List[str], str]] = None
    sampling_params: Optional[Union[List[Dict], Dict]] = None
    
    # 新增字段
    kvpress_params: Optional[Union[List[Dict], Dict]] = None
    # 示例：{"method": "knorm", "compression_ratio": 0.5}
```

**调度器集成：**
```python
class Req:
    def __init__(self, ..., kvpress_params: Optional[Dict] = None):
        self.kvpress_enabled = kvpress_params is not None
        self.kvpress_method = kvpress_params.get("method", "knorm") if kvpress_params else None
        self.kvpress_ratio = kvpress_params.get("compression_ratio", 0.0) if kvpress_params else 0.0
```

#### 挑战 5：Layer-wise 峰值内存

**问题：** `gather` 操作会产生临时内存峰值 (`old_KV + new_KV`)

**解决方案：** 逐层立即释放
```python
for layer_idx in range(num_layers):
    # 压缩当前层
    k_compressed, v_compressed, kept_indices = compress_layer(layer_idx)
    
    # ✅ 立即释放被剪枝的 slot（当前层）
    pruned_indices = get_pruned_for_layer(layer_idx, kept_indices)
    allocator.free(pruned_indices)
    
    # 写入压缩后的 KV
    kv_pool.set_kv_buffer(layer_idx, kept_loc, k_compressed, v_compressed)
    
    # 此时 old_KV 已被释放，峰值受控
```

---

## 集成架构设计

### 整体流程

```
用户请求
    ↓
TokenizerManager
    ├─ 解析 kvpress_params
    └─ 创建 TokenizedGenerateReqInput
    ↓
Scheduler
    ├─ 创建 Req 对象（携带 kvpress 配置）
    └─ 构造 ScheduleBatch
    ↓
prepare_for_extend (Prefill)
    ├─ 检查 req.kvpress_enabled
    ├─ 分配初始 KV cache slot
    └─ 传递 kvpress_ratio 到 ForwardBatch
    ↓
ModelRunner.forward_batch()
    └─ 对每一层：
        ├─ Attention forward
        ├─ 【插入点】kvpress_compress()
        │  ├─ 计算 importance score
        │  ├─ TopK 选择
        │  └─ Gather 压缩
        ├─ allocator.free(pruned_indices)
        └─ set_kv_buffer(compressed_k, compressed_v)
    ↓
prepare_for_decode
    ├─ 使用压缩后的 KV cache
    └─ 每次 decode +1 token（正常流程）
```

### 代码结构

```
python/sglang/srt/
├── kvpress/                          # 新增目录
│   ├── __init__.py
│   ├── base_press.py                 # SGLang 版 BasePress
│   ├── scorer_press.py               # SGLang 版 ScorerPress
│   ├── methods/                      # 具体算法
│   │   ├── knorm_press.py
│   │   ├── snapkv_press.py
│   │   └── expected_attention_press.py
│   └── utils.py                      # 工具函数
│
├── managers/
│   ├── io_struct.py                  # 修改：添加 kvpress_params
│   ├── scheduler.py                  # 修改：处理 kvpress 配置
│   └── schedule_batch.py             # 修改：传递 kvpress_ratio
│
├── layers/attention/
│   └── flashattention_backend.py     # 修改：插入压缩逻辑
│
└── mem_cache/
    └── allocator.py                  # 现有：free() 方法
```

### 核心组件设计

#### 1. SGLangBasePress

```python
@dataclass
class SGLangBasePress:
    """SGLang 版 BasePress，不依赖 PyTorch hook"""
    
    compression_ratio: float = 0.0
    
    def compress(
        self,
        layer_idx: int,
        keys: torch.Tensor,      # (batch, heads, seq_len, dim)
        values: torch.Tensor,    # (batch, heads, seq_len, dim)
        hidden_states: torch.Tensor,  # (batch, seq_len, hidden_dim)
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            compressed_keys: (batch, heads, new_seq_len, dim)
            compressed_values: (batch, heads, new_seq_len, dim)
            kept_indices: (new_seq_len,) 保留的 token 位置
        """
        raise NotImplementedError
```

#### 2. SGLangScorerPress

```python
@dataclass
class SGLangScorerPress(SGLangBasePress):
    """Score-based 压缩方法基类"""
    
    def score(
        self,
        layer_idx: int,
        keys: torch.Tensor,
        values: torch.Tensor,
        hidden_states: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        计算重要性分数
        Returns: scores with shape (batch, heads, seq_len)
        """
        raise NotImplementedError
    
    def compress(self, layer_idx, keys, values, hidden_states, **kwargs):
        if self.compression_ratio == 0:
            return keys, values, torch.arange(keys.shape[2])
        
        # 1. 计算分数
        scores = self.score(layer_idx, keys, values, hidden_states, **kwargs)
        
        # 2. Token-wise 聚合（确保所有 head 删除相同 token）
        scores = scores.mean(dim=1)  # (batch, seq_len)
        
        # 3. TopK 选择
        seq_len = keys.shape[2]
        n_kept = int(seq_len * (1 - self.compression_ratio))
        kept_indices = scores.topk(n_kept, dim=-1).indices
        kept_indices, _ = torch.sort(kept_indices)  # 保持顺序
        
        # 4. Gather 压缩
        # 需要为 heads 维度扩展 indices
        indices_expanded = kept_indices.unsqueeze(1).unsqueeze(-1)  # (batch, 1, n_kept, 1)
        indices_expanded = indices_expanded.expand(-1, keys.shape[1], -1, keys.shape[3])
        
        keys_compressed = keys.gather(2, indices_expanded).contiguous()
        values_compressed = values.gather(2, indices_expanded).contiguous()
        
        return keys_compressed, values_compressed, kept_indices[0]  # 假设 batch=1
```

#### 3. 具体算法：KnormPress

```python
@dataclass
class KnormPress(SGLangScorerPress):
    """Key norm-based 压缩"""
    
    def score(self, layer_idx, keys, values, hidden_states, **kwargs):
        # 计算 key 的 L2 范数（越小越重要）
        return -keys.norm(dim=-1)  # (batch, heads, seq_len)
```

#### 4. 集成函数

```python
# 文件：python/sglang/srt/kvpress/compress.py

def kvpress_compress(
    layer_idx: int,
    keys: torch.Tensor,
    values: torch.Tensor,
    hidden_states: torch.Tensor,
    press_method: str,
    compression_ratio: float,
    **kwargs
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    统一压缩接口
    
    Returns:
        compressed_keys, compressed_values, kept_indices
    """
    # 获取 press 实例（单例模式，避免重复创建）
    press = get_press_instance(press_method, compression_ratio)
    
    return press.compress(
        layer_idx=layer_idx,
        keys=keys,
        values=values,
        hidden_states=hidden_states,
        **kwargs
    )
```

---

## 实现路线图

### Phase 1：基础集成（2 周）
- [ ] 实现 `SGLangBasePress` 和 `SGLangScorerPress`
- [ ] 实现 `KnormPress`（最简单的方法）
- [ ] 修改 `io_struct.py`：添加 `kvpress_params`
- [ ] 修改 `flashattention_backend.py`：插入压缩逻辑
- [ ] 实现 `allocator.free()` 调用

**验证指标：**
- 单请求、单层模型上运行成功
- 显存占用下降（通过 `nvidia-smi` 确认）

### Phase 2：完善功能（2 周）
- [ ] 实现 `SnapKVPress` 和 `ExpectedAttentionPress`
- [ ] 支持 Session 模式（长上下文 + 多问题）
- [ ] 处理 `req_to_token` 映射更新
- [ ] 添加 per-request 配置支持

**验证指标：**
- 多层模型、多请求场景运行
- Session 模式下 context 复用成功

### Phase 3：性能优化（2 周）
- [ ] 优化 gather 操作的内存峰值
- [ ] 支持 Paged allocation 模式
- [ ] 支持 TP 并行（跨 GPU 的 KV 分片）
- [ ] 添加性能监控（压缩耗时、显存节约）

**验证指标：**
- 峰值内存不超过 baseline 的 1.2x
- 压缩耗时 < 5% 的 E2E latency

### Phase 4：评测与发布（1 周）
- [ ] 复现 RULER benchmark
- [ ] 对比 compression_ratio vs accuracy
- [ ] 撰写文档和示例
- [ ] 提交 PR 到 SGLang 主仓库

**验证指标：**
- RULER 4K: compression_ratio=0.5 下，accuracy > 90%
- 显存节约 > 40%

---

## 评测方案

### 评测维度

| 维度 | 指标 | 目标 |
|------|------|------|
| **准确率** | RULER 4K Average Score | > 90% (50% compression) |
| **显存节约** | Peak Memory vs Baseline | > 40% |
| **延迟** | E2E Latency Overhead | < 10% |
| **吞吐** | Requests/sec | 提升（因为显存释放） |

### Benchmark

**1. RULER (4K context)**
- 任务：Needle-in-Haystack, QA, Code, Math
- Metric: String Match Score
- 配置：Llama-3.1-8B, compression_ratio=[0.3, 0.5, 0.7]

**2. 长上下文 QA**
- 数据集：InfiniteBench, LooGLE
- Metric: F1 Score / Exact Match

**3. 性能测试**
- 工具：`notebooks/speed_and_memory.ipynb`（参考 KVPress）
- 监控：Peak Memory, Cache Size, Prefill Time, Decode Time

---

## 风险与缓解

| 风险 | 影响 | 缓解方案 |
|------|------|---------|
| RadixCache 冲突 | 破坏前缀共享 | Session 模式 + 禁用 prefix sharing |
| 峰值内存超限 | OOM | 逐层立即 free() |
| 准确率下降 | 用户不接受 | 提供多种算法选择，可配置 ratio |
| 性能回退 | Latency 增加 | 优化 score 计算，考虑 CUDA kernel |
| TP/PP 不兼容 | 多卡场景失败 | Phase 3 专门处理，先支持单卡 |

---

## 预期成果

### 技术成果
1. **首个** SGLang 上注意力无关的显著显存压缩方案
2. **可插拔架构**：用户可自定义 `CustomPress` 算法
3. **工程化完善**：支持 Session、TP、Paged allocation

### 性能提升
- **显存节约**：30%-70%（取决于 compression_ratio）
- **Decode 加速**：KV cache 变小 → Attention 计算更快（理论提升 10-30%）
- **吞吐提升**：同样显存可服务更多并发请求（提升 50-100%）

### 社区影响
- 为 SGLang 生态提供新的优化维度
- 吸引研究者在 SGLang 上实验新的压缩算法
- 可能成为长上下文推理的标准配置

---

## 参考资料

### KVPress
- 论文：https://arxiv.org/abs/2410.xxxxx（假设存在）
- 代码：https://github.com/NVIDIA/kvpress
- Leaderboard：https://huggingface.co/spaces/nvidia/kvpress-leaderboard

### SGLang
- 文档：https://docs.sglang.ai/
- 代码：https://github.com/sgl-project/sglang
- 博客：https://lmsys.org/blog/2024-01-17-sglang/

### 相关工作
- SnapKV: https://arxiv.org/abs/2404.14469
- StreamingLLM: https://arxiv.org/abs/2309.17453
- H2O (Heavy-Hitter Oracle): https://arxiv.org/abs/2306.14048

---

**最后更新：** 2025-10-16  
**作者：** [Your Name]  
**状态：** 设计阶段

