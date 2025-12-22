#!/usr/bin/env python3
"""
测试原始 test_radix_attention，但使用开放模型 Qwen2.5-1.5B-Instruct
验证我们的 ENCODER_ONLY 修复不会影响生成模型的内存管理
"""

import os
import sys

# 临时修改 DEFAULT_SMALL_MODEL_NAME_FOR_TEST
sys.path.insert(0, '/home/l1q/WSL/sglang/python')
sys.path.insert(0, '/home/l1q/WSL/sglang/test')

# Override the default model
import sglang.test.test_utils as test_utils
test_utils.DEFAULT_SMALL_MODEL_NAME_FOR_TEST = "Qwen/Qwen2.5-1.5B-Instruct"

# Now run the original test
from test.srt.test_radix_attention import *

if __name__ == "__main__":
    os.environ["SGLANG_TEST_RETRACT"] = "true"
    unittest.main()



