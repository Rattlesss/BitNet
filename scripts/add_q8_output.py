"""Add a Q8_0 output.weight tensor to a GGUF by byte-level patching.

Copies the header, KV data, existing tensor infos and tensor data unchanged
(the gguf package cannot size I2_S tensors), appends one tensor info entry and
the Q8_0 data quantized from the F16 token embedding. Original is not modified.
Usage: python scripts/add_q8_output.py SRC.gguf DST.gguf
"""
import hashlib
import struct
import sys

import numpy as np
from gguf import GGMLQuantizationType, GGUFReader, quants

SRC, DST = sys.argv[1], sys.argv[2]
ALIGN = 32
Q8_0_TYPE = 8
CHUNK_ROWS = 4096


def align(n):
    return (n + ALIGN - 1) // ALIGN * ALIGN


r = GGUFReader(SRC)
assert r.alignment == ALIGN
assert "output.weight" not in [t.name for t in r.tensors]
emb = next(t for t in r.tensors if t.name == "token_embd.weight")
assert emb.tensor_type.name == "F16"
n_embd, n_vocab = int(emb.shape[0]), int(emb.shape[1])
assert n_embd % 32 == 0
row_bytes = n_embd // 32 * 34
count = len(r.tensors)

with open(SRC, "rb") as f:
    raw_head = f.read(r.data_offset)
    f.seek(0, 2)
    src_size = f.tell()
assert raw_head[:4] == b"GGUF"
assert struct.unpack_from("<Q", raw_head, 8)[0] == count

pos = r.tensors[0].field.offset
for _ in range(count):
    name_len = struct.unpack_from("<Q", raw_head, pos)[0]
    pos += 8 + name_len
    n_dims = struct.unpack_from("<I", raw_head, pos)[0]
    pos += 4 + 8 * n_dims + 4 + 8
info_end = pos
assert info_end <= r.data_offset
assert align(info_end) == r.data_offset
assert set(raw_head[info_end:]) <= {0}

old_data_len = src_size - r.data_offset
new_rel = align(old_data_len)
pad = new_rel - old_data_len

name = b"output.weight"
entry = (struct.pack("<Q", len(name)) + name + struct.pack("<I", 2)
         + struct.pack("<QQ", n_embd, n_vocab) + struct.pack("<I", Q8_0_TYPE)
         + struct.pack("<Q", new_rel))
new_head = bytearray(raw_head[:info_end])
struct.pack_into("<Q", new_head, 8, count + 1)
new_head += entry
new_data_offset = align(len(new_head))
new_head += b"\x00" * (new_data_offset - len(new_head))
expected = new_data_offset + new_rel + n_vocab * row_bytes
print("expected size", expected)

with open(SRC, "rb") as src, open(DST, "wb") as dst:
    dst.write(new_head)
    src.seek(r.data_offset)
    while True:
        b = src.read(64 << 20)
        if not b:
            break
        dst.write(b)
    dst.write(b"\x00" * pad)
    mm = np.memmap(SRC, dtype=np.float16, mode="r", offset=emb.data_offset,
                   shape=(n_vocab, n_embd))
    for i in range(0, n_vocab, CHUNK_ROWS):
        blk = np.asarray(mm[i:i + CHUNK_ROWS], dtype=np.float32)
        q = quants.quantize(blk, GGMLQuantizationType.Q8_0)
        assert q.shape[-1] == row_bytes
        dst.write(q.tobytes())
    assert dst.tell() == expected
print("written", DST)


def digest(path, start, length):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        f.seek(start)
        left = length
        while left:
            b = f.read(min(left, 64 << 20))
            assert b
            h.update(b)
            left -= len(b)
    return h.hexdigest()


rd = GGUFReader(DST)
assert rd.data_offset == new_data_offset
assert len(rd.tensors) == count + 1
src_rel = {t.name: t.data_offset - r.data_offset for t in r.tensors}
for t in rd.tensors:
    if t.name == "output.weight":
        assert t.tensor_type.name == "Q8_0"
        assert [int(x) for x in t.shape] == [n_embd, n_vocab]
        assert t.data_offset - rd.data_offset == new_rel
    else:
        assert t.data_offset - rd.data_offset == src_rel[t.name], t.name
with open(SRC, "rb") as a, open(DST, "rb") as b:
    assert a.read(8) == b.read(8)
    a.seek(16)
    b.seek(16)
    assert a.read(info_end - 16) == b.read(info_end - 16)
assert digest(SRC, r.data_offset, old_data_len) == digest(DST, new_data_offset, old_data_len)
print("header, KV, tensor infos and old tensor data identical")

qmm = np.memmap(DST, dtype=np.uint8, mode="r", offset=new_data_offset + new_rel,
                shape=(n_vocab, row_bytes))
for i in (0, n_vocab // 2, n_vocab - 1):
    ref = np.asarray(mm[i], dtype=np.float32)
    deq = quants.dequantize(np.asarray(qmm[i:i + 1]), GGMLQuantizationType.Q8_0)[0]
    err = float(np.abs(deq - ref).max())
    assert err <= float(np.abs(ref).max()) / 127 + 1e-6, (i, err)
    print("row", i, "max abs error", err)
print("OK")
