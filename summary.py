#!/usr/bin/env python3
"""summary.py <results dir> <box> <stamp> <ids,comma>  - the ladder as a table, from the
results JSONs written since <stamp>."""
import json, sys, glob, os, collections
d, box, stamp, models = sys.argv[1:5]
rows = []; unsub = 0
for f in sorted(glob.glob(f"{d}/{box}_*.json")):
    if os.path.basename(f)[len(box) + 1:-5] < stamp: continue
    j = json.load(open(f)); rows += j.get("rows", []); unsub += len(j.get("unsubmitted", []))
cell = collections.defaultdict(dict)
for r in rows: cell[(r["model"], r["scenario"])][r["metric"]] = r["value"]
rungs = ["4k", "16k", "32k", "64k", "128k", "256k", "4k_x4"]
print(f"suite v3 on {box}, {stamp}: {len(rows)} rows, {unsub} failed to post\n")
print(f"{'model':42}" + "".join(f"{r:>18}" for r in rungs))
print(f"{'':42}" + "".join(f"{'pp / tok/s / ttft':>18}" for _ in rungs))
for m in models.split(","):
    line = f"{m:42}"
    for r in rungs:
        c = cell.get((m, r), {})
        f = lambda k, fmt: (fmt % c[k]) if k in c else "-"
        if r == "4k_x4": s = f"{f('gen_tok_s', '%.1f')} / {f('gen_tok_s_aggregate', '%.0f')}" if c else "-"
        else: s = f"{f('prompt_tok_s', '%.0f')} / {f('gen_tok_s', '%.1f')} / {f('ttft_s', '%.0fs')}" if c else "-"
        line += f"{s:>18}"
    print(line)
