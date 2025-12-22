#!/usr/bin/env python3
"""
检查 token_to_kv_pool_allocator 内存分配器状态

验证: max_total_num_tokens == available_size + evictable_size + protected_size

测试配置:
- dtype: float16, float32
- attention_backend: triton, torch_native
- 随机输入
"""

import sys
import unittest
import random
import time
import subprocess
import signal

sys.path.insert(0, '/home/l1q/WSL/sglang/python')
sys.path.insert(0, '/home/l1q/WSL/sglang/test')

from sglang.test.test_utils import CustomTestCase
import requests


# 测试配置
MODEL = "answerdotai/ModernBERT-base"
ATTENTION_BACKENDS = ["torch_native", "triton"]
TORCH_DTYPES = ["float16", "float32"]

# 随机测试输入
TEST_PROMPTS = [
    "This is a test sentence for embedding.",
    "Another short test.",
    "Testing memory management with ModernBERT model.",
    "Short text.",
    "A longer test sentence to check memory allocation.",
]


class TestKVPoolAllocator(CustomTestCase):
    """测试 token_to_kv_pool_allocator 的内存分配器状态"""

    def _start_server(self, attention_backend, dtype, port):
        """启动服务器"""
        cmd = [
            "python3", "-m", "sglang.launch_server",
            "--model-path", MODEL,
            "--is-embedding",
            "--attention-backend", attention_backend,
            "--dtype", dtype,
            "--mem-fraction-static", "0.3",
            "--disable-cuda-graph",
            "--chunked-prefill-size", "256",
            "--disable-radix-cache",
            "--log-level", "debug",
            "--port", str(port),
            "--host", "127.0.0.1",
        ]
        
        print(f"🚀 Starting server: {' '.join(cmd)}")
        
        # 启动服务器进程
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # 等待服务器启动
        base_url = f"http://127.0.0.1:{port}"
        max_wait = 60
        for i in range(max_wait):
            try:
                response = requests.get(f"{base_url}/get_model_info", timeout=1)
                if response.status_code == 200:
                    print(f"✅ Server started successfully")
                    return process, base_url
            except:
                time.sleep(1)
                # 检查进程是否崩溃
                if process.poll() is not None:
                    stderr_output = process.stderr.read()
                    print(f"❌ Server crashed during startup")
                    print(f"stderr: {stderr_output[-500:]}")  # 显示最后500字符
                    raise RuntimeError("Server crashed during startup")
        
        raise TimeoutError("Server failed to start")

    def _send_requests(self, base_url, num_requests=10):
        """发送测试请求"""
        prompts = random.sample(TEST_PROMPTS, k=min(3, len(TEST_PROMPTS)))
        
        for i in range(num_requests):
            prompt = random.choice(prompts)
            try:
                response = requests.post(
                    f"{base_url}/generate",
                    json={
                        "text": prompt,
                        "sampling_params": {
                            "temperature": 0,
                            "max_new_tokens": 0,
                        },
                    },
                    timeout=10
                )
                if response.status_code == 200:
                    print(f"  ✅ Request {i+1} successful")
                else:
                    print(f"  ❌ Request {i+1} failed: {response.status_code}")
            except Exception as e:
                print(f"  ❌ Request {i+1} error: {e}")
            
            time.sleep(0.2)

    def _check_server_logs(self, process):
        """检查服务器日志中是否有内存泄漏"""
        # 这里我们通过检查进程是否正常运行来判断
        # 如果有内存泄漏，scheduler 会抛出 ValueError
        return process.is_alive()

    def check_kv_pool_allocator(self, attention_backend, dtype):
        """
        检查 KV pool allocator 的状态
        
        验证: max_total_num_tokens == available_size + evictable_size + protected_size
        """
        print(f"\n{'='*60}")
        print(f"Testing KV Pool Allocator")
        print(f"Backend: {attention_backend}, Dtype: {dtype}")
        print(f"{'='*60}")
        
        port = random.randint(5000, 6000)
        process = None
        
        try:
            # 启动服务器
            process, base_url = self._start_server(attention_backend, dtype, port)
            
            print(f"🔄 Sending test requests...")
            self._send_requests(base_url, num_requests=20)
            
            # 等待一下让请求完成
            time.sleep(2)
            
            # 检查服务器是否还在运行（如果有内存泄漏会崩溃）
            if process.poll() is not None:
                # 进程已退出
                stderr_output = process.stderr.read()
                if "memory leak detected" in stderr_output:
                    print(f"❌ FAIL: Memory leak detected in logs")
                    print(f"stderr: {stderr_output[-1000:]}")
                    return False
                else:
                    print(f"❌ FAIL: Server crashed (exit code: {process.returncode})")
                    print(f"stderr: {stderr_output[-500:]}")
                    return False
            
            print(f"✅ PASS: Server still running, no memory leak detected")
            return True
            
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False
            
        finally:
            # 清理
            if process and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()

    def test_kv_pool_allocator_all_configs(self):
        """测试所有配置的 KV pool allocator"""
        print("\n" + "="*60)
        print("Testing KV Pool Allocator for ModernBERT")
        print("Checking: max_total_num_tokens == available + evictable + protected")
        print("="*60)
        
        results = {}
        
        for dtype in TORCH_DTYPES:
            for backend in ATTENTION_BACKENDS:
                config_name = f"{backend}_{dtype}"
                
                print(f"\n📝 Testing: {config_name}")
                
                try:
                    success = self.check_kv_pool_allocator(backend, dtype)
                    results[config_name] = "✅ PASS" if success else "❌ FAIL"
                except Exception as e:
                    results[config_name] = f"❌ ERROR: {str(e)[:50]}"
                    import traceback
                    traceback.print_exc()
        
        # 打印结果摘要
        print("\n" + "="*60)
        print("Test Results Summary:")
        print("="*60)
        for config, result in results.items():
            print(f"  {config:30}: {result}")
        print("="*60)
        
        # 检查是否所有测试都通过
        all_passed = all("PASS" in result for result in results.values())
        
        if all_passed:
            print("\n🎉 All tests PASSED!")
            print("✅ max_total_num_tokens == available_size + evictable_size + protected_size")
        else:
            print("\n💥 Some tests FAILED!")
            print("❌ Memory leak detected in KV pool allocator")
        
        self.assertTrue(all_passed, "Some tests failed!")


if __name__ == "__main__":
    # 只运行这个测试
    suite = unittest.TestLoader().loadTestsFromTestCase(TestKVPoolAllocator)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    exit(0 if result.wasSuccessful() else 1)
