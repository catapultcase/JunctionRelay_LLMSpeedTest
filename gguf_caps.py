"""What capabilities do these weights ACTUALLY have?

Reads the architecture and the tensor-name table straight out of the GGUF
header and reports whether MTP / next-token-prediction tensors are present.

⛔ WHY THIS EXISTS. The house rule is to bench with every capability the weights
have - which means knowing what they have. That was being taken from prose: a
catalog trait, a comment counting tensors, a note someone wrote once. On
2026-08-28 that cost most of a day. Flash-Next was given --spec-type draft-mtp
because the CATALOG said the checkpoint has MTP; its UD-IQ4_XS quant does not,
llama-server aborted the load rather than ignoring the flag, and every run came
back as a bare HTTP 500 that was read as a VRAM cliff.

A trait describes a CHECKPOINT. A quant is not its checkpoint. Ask the file.

    python3 gguf_caps.py model.gguf [more.gguf ...]

⚠️ For a split GGUF, pass EVERY shard. The first shard can declare 0 tensors -
it carries metadata - so scanning it alone proves nothing either way.

Verified against known answers 2026-08-28: qwen3.8-27b NVFP4 has 4 nextn
tensors (and MTP works), qwen3.6-35b-a3b 733 and qwen3-coder-next 843 have
none, Flash-Next has none across 1224 tensors in shards 2-3.
"""
import struct, sys

def scan(path):
    f = open(path, "rb")
    magic = f.read(4)
    if magic != b"GGUF":
        return path, "NOT GGUF", []
    ver, = struct.unpack("<I", f.read(4))
    n_tensors, = struct.unpack("<Q", f.read(8))
    n_kv, = struct.unpack("<Q", f.read(8))

    def rd_str():
        n, = struct.unpack("<Q", f.read(8))
        return f.read(n).decode("utf-8", "replace")

    def skip_val(t):
        sizes = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
        if t == 8:
            rd_str()
        elif t == 9:
            et, = struct.unpack("<I", f.read(4))
            n, = struct.unpack("<Q", f.read(8))
            for _ in range(n):
                skip_val(et)
        else:
            f.read(sizes[t])

    arch = None
    for _ in range(n_kv):
        k = rd_str()
        t, = struct.unpack("<I", f.read(4))
        if k == "general.architecture":
            arch = rd_str()
        else:
            skip_val(t)

    names = []
    for _ in range(n_tensors):
        nm = rd_str()
        nd, = struct.unpack("<I", f.read(4))
        f.read(8 * nd)
        f.read(4)   # type
        f.read(8)   # offset
        names.append(nm)
    hits = [n for n in names if any(x in n.lower() for x in ("mtp", "nextn", "eh_proj", "draft"))]
    return path, f"{arch} | {n_tensors} tensors", hits

for p in sys.argv[1:]:
    try:
        path, info, hits = scan(p)
        print(f"{path.split('/')[-1]}\n  {info}\n  MTP-ish tensors: {len(hits)}"
              + (f" -> {hits[:4]}" if hits else " -> NONE"))
    except Exception as e:
        print(f"{p}: ERROR {e}")
