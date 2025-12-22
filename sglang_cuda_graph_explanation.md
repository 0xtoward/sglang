# SGLang 的 CUDA Graph 机制详解

## 1. 什么是 CUDA Graph？

CUDA Graph 是 NVIDIA 提供的一种性能优化技术，核心思想是：
- **普通方式**：每次 forward 都从 CPU 提交一堆 CUDA kernel 到 GPU（高延迟）
- **CUDA Graph**：把整个 forward pass 的 kernel 序列"录制"下来，后续直接在 GPU 上重放（低延迟）

### 性能收益
- 减少 CPU-GPU 通信开销（~10-30% 加速）
- 特别适合 **decode 阶段**（因为每次只生成 1 个 token，kernel 调用开销占比大）

### 限制
- **输入形状必须固定**：录制时的 batch size / seq_len 必须与重放时完全一致
- **不能有动态控制流**：if/else 分支、动态内存分配等都会导致 graph 失效

---

## 2. SGLang 如何使用 CUDA Graph？

### 2.1 核心文件
- **`python/sglang/srt/model_executor/model_runner.py`**：ModelRunner 负责 forward 和 CUDA Graph 的录制/重放
- **`python/sglang/srt/layers/attention/triton_backend.py`**（或 `flashinfer_backend.py`）：Attention 层需要支持 CUDA Graph 模式

### 2.2 录制流程（Capture）

```python
# python/sglang/srt/model_executor/model_runner.py

def capture_cuda_graphs(self):
    """在初始化时录制 CUDA Graph，针对不同的 batch size"""
    
    # 1. 禁用 CUDA Graph 模式，使用普通模式运行一次 warmup
    self.enable_cuda_graph = False
    dummy_batch = self._create_dummy_batch(batch_size=1)
    self.forward(dummy_batch)  # Warmup
    
    # 2. 对每个常见的 batch size 录制 graph
    for bs in [1, 2, 4, 8, 16]:
        # 创建固定形状的 dummy batch
        dummy_batch = self._create_dummy_batch(batch_size=bs)
        
        # 开始录制
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            logits = self.forward(dummy_batch)  # 录制所有 kernel 调用
        
        # 保存 graph 和对应的输入/输出 tensor
        self.cuda_graphs[bs] = {
            'graph': graph,
            'input_ids': dummy_batch.input_ids,  # 保存输入 tensor 的引用
            'seq_lens': dummy_batch.seq_lens,
            'logits': logits,  # 保存输出 tensor 的引用
        }
```

**关键点**：
- 录制时，PyTorch 会记录所有 CUDA kernel 的调用序列和参数
- **输入/输出 tensor 的内存地址被固定**：后续重放时必须写入同样的地址

---

### 2.3 重放流程（Replay）

```python
def forward(self, batch: ForwardBatch):
    bs = len(batch.reqs)
    
    # 检查是否可以使用 CUDA Graph
    if self.enable_cuda_graph and bs in self.cuda_graphs:
        graph_data = self.cuda_graphs[bs]
        
        # 1. 把真实数据拷贝到录制时的 input tensor
        graph_data['input_ids'].copy_(batch.input_ids)
        graph_data['seq_lens'].copy_(batch.seq_lens)
        graph_data['req_pool_indices'].copy_(batch.req_pool_indices)
        # ... 拷贝其他输入
        
        # 2. 重放 graph（所有 kernel 直接在 GPU 上执行，无 CPU 开销）
        graph_data['graph'].replay()
        
        # 3. 从录制时的 output tensor 读取结果
        logits = graph_data['logits']
        return logits
    else:
        # 普通模式：逐个提交 kernel
        return self._forward_impl(batch)
```

**关键限制**：
- **batch size 必须匹配**：如果当前 batch size 没有对应的 graph，回退到普通模式
- **seq_lens 必须固定**：如果 `seq_lens` 的形状或值变化（比如 KVPress 压缩后），graph 失效

---

### 2.4 Attention Backend 的适配

```python
# python/sglang/srt/layers/attention/triton_backend.py

def init_forward_metadata(self, forward_batch: ForwardBatch):
    """普通模式：每次都重新计算 kv_indices"""
    bs = len(forward_batch.reqs)
    
    # 从 req_to_token 构建 kv_indices（需要访问 forward_batch.reqs）
    kv_indices = self._build_kv_indices(forward_batch.reqs, forward_batch.seq_lens)
    
    # 准备 attention 输入
    self.kv_indptr = compute_indptr(forward_batch.seq_lens)
    self.kv_indices = kv_indices

def init_forward_metadata_replay_cuda_graph(self, bs: int, req_pool_indices: torch.Tensor):
    """CUDA Graph 模式：只能用固定的 tensor，无法访问 forward_batch.reqs"""
    
    # ❌ 不能访问 forward_batch.reqs（因为 graph 录制时没有真实 reqs）
    # ✅ 只能用录制时的固定 tensor（kv_indptr, kv_indices 等）
    
    # 使用预先分配的 buffer
    self.kv_indptr = self.cuda_graph_kv_indptr[bs]
    self.kv_indices = self.cuda_graph_kv_indices[bs]
```

**关键区别**：
- **普通模式**：可以读取 `forward_batch.reqs`，动态计算 `kv_indices`
- **CUDA Graph 模式**：**无法访问 `reqs`**，只能用录制时的固定 tensor

---

## 3. KVPress 与 CUDA Graph 的冲突

### 3.1 冲突点 1：`seq_lens` 变化

```python
# KVPress 压缩后
req.actual_kv_len = 70  # 原本 100 tokens，压缩到 70

# Decode 时
batch.seq_lens[i] = req.actual_kv_len + len(req.output_ids)  # 动态变化！
```

**问题**：`seq_lens` 的值变了，但 CUDA Graph 录制时假定它是固定的！

---

### 3.2 冲突点 2：`req_to_token` 更新

```python
# KVPress 压缩后
self.req_to_token_pool.req_to_token[req_pool_idx, :n_kept] = kv_indices[:n_kept]
self.req_to_token_pool.req_to_token[req_pool_idx, n_kept:seq_len] = 0  # 后面置零
```

**问题**：`req_to_token` 变了，但 CUDA Graph 中的 `kv_indices`（从 `req_to_token` 计算而来）是固定的！

---

### 3.3 冲突点 3：无法在 CUDA Graph 模式下调用压缩逻辑

```python
def init_forward_metadata_replay_cuda_graph(self, bs: int, req_pool_indices: torch.Tensor):
    # ❌ 无法访问 forward_batch.reqs
    # ❌ 无法调用 _kvpress_compress_single_req(req)
    
    # 即使我们想在这里压缩，也拿不到 req 对象！
```

---

## 4. 我们的解决方案：先禁用 CUDA Graph

### 4.1 临时方案

在 `launch.json` 中添加 `--disable-cuda-graph`，跳过 CUDA Graph 的录制和使用。

```json
{
    "args": [
        "--disable-cuda-graph",  // ← 强制走普通模式
        "--enable-kvpress",
        "--kvpress-compression-ratio", "0.1"
    ]
}
```

**优点**：
- KVPress 逻辑不受任何限制
- 可以动态修改 `seq_lens` 和 `req_to_token`

**缺点**：
- 失去 CUDA Graph 的性能优化（~10-30% 的加速）

---

### 4.2 长期方案：让 KVPress 兼容 CUDA Graph

需要解决以下问题：

#### 方案 A：引入 `physical_seq_lens`（我们刚刚撤销的）
- **Prefill** 阶段：压缩后更新 `req.actual_kv_len`
- **Decode** 阶段：计算 `physical_seq_lens`（用于 attention），但保持 `seq_lens` 不变（用于 CUDA Graph 的形状匹配）
- **问题**：需要修改所有 Attention Backend，且 CUDA Graph 模式下无法访问 `req.actual_kv_len`

#### 方案 B：预先录制"压缩后的 CUDA Graph"
- 在录制 graph 时，假设 KV cache 已经压缩到特定长度（如 70%）
- Decode 时，如果实际压缩率匹配，直接重放对应的 graph
- **问题**：需要为每个压缩率录制单独的 graph，内存开销大

#### 方案 C：放弃 CUDA Graph（推荐）
- KVPress 的收益（显存压缩 + 更大 batch size）**远大于** CUDA Graph 的 10-30% 加速
- 特别是当 batch size 增大后，kernel 调用开销占比降低，CUDA Graph 的收益也会减少

---

## 5. 总结

### CUDA Graph 的核心特点
1. **录制-重放**：提前记录 kernel 序列，减少 CPU-GPU 通信
2. **固定形状**：输入/输出 tensor 的形状和地址必须不变
3. **无动态逻辑**：不能有 if/else、动态内存分配等

### KVPress 与 CUDA Graph 的矛盾
1. KVPress 会修改 `seq_lens`（从 100 → 70）
2. KVPress 会修改 `req_to_token`（置零后半部分）
3. CUDA Graph 模式下无法访问 `reqs`，无法执行压缩逻辑

### 当前策略
- **短期**：禁用 CUDA Graph（`--disable-cuda-graph`），让 KVPress 正常工作
- **长期**：评估是否需要兼容 CUDA Graph，或接受"KVPress 与 CUDA Graph 互斥"的现状

---

## 附录：关键代码位置

| 功能 | 文件 | 关键函数 |
|------|------|---------|
| CUDA Graph 录制 | `model_runner.py` | `capture_cuda_graphs()` |
| CUDA Graph 重放 | `model_runner.py` | `forward()` 中的 `if self.enable_cuda_graph` 分支 |
| Attention 普通模式 | `triton_backend.py` | `init_forward_metadata()` |
| Attention Graph 模式 | `triton_backend.py` | `init_forward_metadata_replay_cuda_graph()` |
| 启用/禁用开关 | `server_args.py` | `--disable-cuda-graph` |


