"""List non-block tensors and block 0 tensors of a GGUF file.

Note: the gguf package does not know I2_S, so n_bytes is overcounted 4x for
I2_S tensors (it assumes 1 byte per element). F16/F32 sizes are correct.
"""
import sys
from gguf import GGUFReader

path = sys.argv[1] if len(sys.argv) > 1 else "models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"
for t in GGUFReader(path).tensors:
    if not t.name.startswith("blk.") or t.name.startswith("blk.0."):
        print(t.name, t.tensor_type.name, list(t.shape), t.n_bytes)
