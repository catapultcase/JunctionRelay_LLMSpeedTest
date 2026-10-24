# LLMSpeedTest

A speed suite for JunctionRelay's benchmark contract (`docs/BENCHMARK-CONTRACT.md` in JR).
Runs from one coordinating machine, measures each box in `targets.json`, and posts every row
to JR over MCP as it is measured, as **JR Speed Test** (`suite_name`; JR keeps no table of
suite names).

## The suite (version 5, 2026-10-24)

**Scenarios are prompt depths.** The ladder is one cold request per rung at tg128; the
concurrency rung is its own scenario, driven long enough to overlap (v5):

| scenario | prompt | metrics |
|---|---|---|
| `4k` `16k` `32k` `64k` `128k` `256k` | the frozen corpus cut to that many tokens (cycled past 51.4K), salted, `cache_prompt=false` | `prompt_tok_s` (prefill of the whole prompt), `gen_tok_s` (next 128 tokens), `ttft_s` (server `prompt_ms`), `draft_accept`, `gpu_watts_avg` |
| `<depth>_xN` (v5; the deepest rung that fits, 32K target / 4K floor) | N streams at that depth, tg1024 | `gen_tok_s` (mean per stream, with its overlap factor and a WARNING note when the streams never overlapped), `gen_tok_s_aggregate` |
| — | after the run | `vram_gb`; `disk_read_mbs` and `vram_shared_gb` on the deepest rung |

Every rung the window allows runs (`k·1000 + 640 ≤ ctx`). `--quick` = 4k, 32k, 128k;
`--depths 4,32` for a custom set. The matrix shows 4k / 32k / 128k; the rest ride the
reports. This is the shape published numbers take (`llama-bench -d`, the vLLM/MLX recipes).

v1–v2 (`shallow` / `deep` / `review` / `parallel_x4`) are retired with their history. v3 was this
ladder on a private corpus; v4 swapped in a public one (`corpus/README.md`), so any lab's cells
compare with any other's. v5 rebuilt the concurrency rung only: the ladder above is unchanged, but
v4's `4k_x4` cells were driven at tg128 against a ~9 s prefill queue and are not comparable to
v5's sustained `<depth>_xN` runs.

## Setup

| What | Where |
|---|---|
| Coordinator | Any machine that reaches the boxes' APIs and ssh. It is never a target itself |
| Targets | `targets.json` beside `bench.py` (gitignored; `JR_SPEED_TARGETS` names another path). Copy `targets.example.json`; the fields are explained above `load_targets()` in `bench.py` |
| Engines | llama-swap in front of llama-server, or a container engine (vLLM, TensorFold) with Prometheus `/metrics`, set as `"engine": "vllm"` |
| Ledger | JR's MCP endpoint: `JR_MCP_URL` + `JR_MCP_TOKEN`, else the `lab-dev` server in `~/.openclaw/openclaw.json` (otago), else the box's own OpenCode client config's `junction-relay` entry (fw13) |

## Running

```
nohup python3 bench.py --all > bench.log 2>&1 &     # every model, every box
python3 bench.py --target <box> --models <id,id> [--quick] [--dry-run] [--llama-bench]
python3 bench.py --gap-fill --target <box> --models <other box's id>
python3 bench.py --entries <file>.json --target <box>     # {id: launch cmd} placements
```

Detach it: it posts as it goes. One run per box (`results/.lock-<box>`); different boxes
run concurrently. Rows that fail to post stay in `results/<box>_<stamp>.json`. Selector
ids (`brain`) are skipped.

`--llama-bench` adds, after each model's ladder, `llama-bench -p 512 -n 128 -d 4096,32768,131072`
with the same file, `-ot` placement, split, KV type, ubatch and load mode (the server is
unloaded for it, the next request reloads). Rows post as `llb_pp512` / `llb_tg128` under the
depth's scenario - the number a llama.cpp thread posts, beside ours. Its tg128 never has the
draft head: llama-bench has no speculation flag.

`summary.py <results> <box> <stamp> <ids>` prints a run's table. Unattended drivers (wait for
idle, bench every served model, reload the daily) belong to the lab that runs them.

## What a row carries

Setup from the running server, never the catalog: `ctx`/`parallel`/`engine`/`vision`
from `/props`; `spec_type`, `cache_k/v`, quant (from `model_path`) and the **placement**
from the `/running` launch cmd — `n_cpu_moe` (the `--n-cpu-moe` count **plus** every
`-ot …exps=CPU` pin), `load_mode`, `tensor_split`, `tensor_overrides`, `ubatch`.
`weights_gb` and `vram_gb` are read over ssh. Speeds come from the completion's
`timings` block (activity-log fallback); wall clock is never a tok/s.

## vLLM and TensorFold targets (2026-10-01)

`"engine": "vllm"` on the target (TensorFold included: `engine_of()` reads its
`/metrics`, which publishes `tensorfold:*`). Same ladder, same metrics, different sources:

| | llama-server | vLLM |
|---|---|---|
| speeds | the completion's `timings` | `/metrics` deltas across the request: `request_prefill_time` (scheduled → first token), `request_decode_time` (first → last token), `spec_decode_*` |
| setup | `/props` + the `/running` cmd | the container's launch argv (`docker inspect` over ssh), `/version`, `/v1/models` |
| GPU proof | `llama-server` in `nvidia-smi` | `VLLM::Worker` in `nvidia-smi` |
| cold prefill | salt + `cache_prompt=false` | salt; the template's system block still hits the prefix cache, so `prompt_tok_s` is over the uncached tokens and the note carries `cache_n` |

- ⛔ **The box must be idle.** A delta is one request's only if nothing else finished inside it; the run waits for
  idle, checks `request_success` moved by exactly its own count, and retries with a fresh salt (3 tries) otherwise.
- `gen_tok_s` is `(tokens − 1) / decode time`; on `<depth>_xN` it is summed tokens over summed decode time, and the row's note carries the overlap factor.
- The top rung is cut with `/tokenize` to land under the window: vLLM refuses an over-long prompt, it does not truncate.
- `gpu_watts_avg` on a Spark is the GPU rail only, about half the box.
- The quant carries its publisher when the weights are not the vendor's own (`NVFP4-Mia` for
  `Mia-AiLab/...-NVFP4`; `VENDOR_ORGS` lists the orgs that post bare). `ctx` is the window one request
  gets (`--max-model-len`), not a pool divided by slots.
- No `--gap-fill`, `--entries`, `--llama-bench` or `wont_fit`: there is no yaml and no load to refuse.
- **TensorFold's own facts**, read from the container argv: `--kv-dtype` when the recipe passes one
  (Flash-Next `bf16`; GLM `fp8` via `TF_GLM_KV`), else **`native`** - the engine takes no KV choice and
  uses the model's own layout (DeepSeek-V4.1's MLA latent cache; ⛔ not f16). Thinking is `--thinking` /
  `--no-thinking`; `--reasoning-effort` is recorded when the launch states one, else the row says `on`
  (enabled, no effort sent). The contract's vocabulary section defines both terms for every engine.

## A rung that dies (2026-09-25)

A ladder rung that crashes the server is a RESULT, not a gap. `bench_model` posts **`rung_failed`**
(value 1, `notes` = the cause line from the box's own log: cudaMalloc / allocation / kernel OOM killer /
segfault, via `rung_cause()`) and **stops the ladder**: every later rung and the concurrency step would go
to a dead or reloading server and fail with no cause of their own. JR renders FAILED on that rung and
"not reached" after it; a rung deeper than the window is N/A and posts nothing. No cause line in the log,
nothing is recorded (the same evidence gate as a refusal).

Re-bench ids may carry a `-ramNN` placement tag and a `-ubNNN` batch tag (the ub sweep); `catalog_name()`
strips both, so they never mint a phantom model. Split, `-ub` and the `-ot` placement are row identity
in JR since 0.9.0.180 - a different one is a different row, never a newer number for the same row.

## Gap-fill and re-bench (temp entries)

`--gap-fill` benches a setup another box serves: reads the other box's master
(the target's `master` in `targets.json`, a copy kept current), rewrites paths and
split macros, appends a temp entry to the target's **deployed** yaml, benches (or records
`wont_fit` with the load error from the box's own logs), then **restores the exact pre-run
yaml from its snapshot** (`results/.yaml-before-<box>-<stamp>`) and reloads the box's daily
model. It refuses when the id is already served there, no master has it, the weights are
missing, or the setup spills experts on a zero-offload box.

`--entries <json>` does the same for a `{id: launch cmd}` file - the box's own retired
placements, their commands read from the journal's `Executing start command` lines. An id
that collides with a served name gets a suffix (`-ram37`); the row key ignores the id.

## Rules

- ⛔ **Never CPU-only.** After every load the serving process must appear in `nvidia-smi`'s
  compute list over ssh, or the run aborts. No override.
- ⛔ **Bench with every capability the QUANT has** — MTP where the file carries the head
  (`gguf_caps.py <shards>`), mmproj where it ships one.
- ⛔ **Tested data only.** A `wont_fit` needs a load error in the box's logs; otherwise
  the cell stays untested.
- ⛔ **Never refresh the corpus in place** — a changed corpus is a changed scenario; bump
  `SUITE_VERSION` with it.
- The coordinator is never a target. The deployed yaml is never the master.

## Lessons learned

- **`gen_tok_s_aggregate` is output over wall clock INCLUDING prefill.** At a 4K prompt it was mostly
  prefill time, and read as "what the box writes with four streams" it said Spark A does not scale
  (41.5 against 40 alone) and a wrong rule went into OpenCode's `AGENTS.md` (2026-10-01). The v5
  concurrency rung drives tg1024 at the deepest rung that fits, so prefill is a smaller share and the
  row carries its overlap factor. **Read both figures together:** per stream, `each` is the rate a
  stream gets while the run decodes; across the box, `total` is what four sessions receive together.
- **A short decode window measures the scheduler, not the box.** v4's `4k_x4` (tg128) spent ~2 s
  decoding per stream against ~9 s of serialized 4k prefills. On the coding pair that made every
  stream decode nearly alone: `each` read 67.3 - its solo speed - inside a 4-stream test, while the
  total was 30.3. `each > total` is the tell; the v5 row now states the overlap and warns
  (2026-10-24).
- **The corpus is real text for a reason.** Random filler measures the same prefill but
  says nothing about generation at depth; the review ask at the end keeps the 128
  generated tokens honest.
- **Cold means cold.** llama-server's `--cache-reuse` splices matching *chunks* back in
  even after the prefix diverges; the salt goes first *and* the request carries
  `cache_prompt=false`. A rung with `cache_n > 0` is noted on the row, not trusted.
- **The log must be tailed deep and stripped of the suite's own polling** before the
  cause of a failure is read — within seconds of a failure the last 200 lines are
  `/running` polls (2026-08-28, both boxes: real OOMs recorded as nothing).
- **cheyenne logs to the journal**, not `/opt/llm/logs` (empty). Reading the empty dir
  made every refusal there INCONCLUSIVE (2026-08-28).
- **`/props` does not report speculation** (`draft-mtp` shows `speculative.types: none`
  on b10360 and b10639) — read the launch cmd.
- **The quant, not the checkpoint.** Flash-Next's UD-IQ4_XS has no MTP layers; given
  `--spec-type draft-mtp` llama-server aborts and every run read as a VRAM cliff
  (2026-08-28). Ask the file.
- **A pin to CPU is offload.** Counting `--n-cpu-moe` alone filed a 37-in-RAM placement as
  27 and a name-pinned one as all-in-VRAM (2026-09-24).
- **The `--n-cpu-moe` cliff is silent** — sweep down until throughput collapses, back off
  one (Flash-Next on pris: 30.06 tok/s at 29, 7.69 at 28). `--load-mode none` is required
  with it (mmap + CPU pins collapses prefill, 2026-08-15).
- **`errors="replace"` on every subprocess read** — pris's console is not UTF-8 and one
  byte killed a run (2026-08-28).
- **Windows boxes page silently**: `vram_shared_gb` ≥ 0.8 GB on a no-offload setup is
  a regime of its own (0.65 GB full speed, 1.1 GB −35% gen, 4.9 GB −93%).
- **Concurrent runs on different boxes do not interfere** (pris 143.04 alone vs 143.64
  with cheyenne mid-run, 2026-08-28); a second run on the same box does.
- **A deep rung can OOM the HOST, not the card.** Flash-Next (37 layers pinned in RAM) at the
  256k rung was killed by the kernel OOM killer and took llama-swap down; systemd restarted
  it and the preload restored the daily (2026-09-25). The ladder survives it: the rung is
  skipped, the next model's warm-up reloads, but the same model's `4k_x4` runs against a
  dead server and reads "no clean streams" - re-run it with `--depths 4` if it matters.
- **The journal's launch line ends in `, env: [...]`.** Deployed verbatim, llama-server
  refused every entry with `invalid argument: env:` - and because "error" matched the
  refusal filter, eight bogus `wont_fit` verdicts were recorded before the run was stopped
  (2026-09-25). The tail is stripped, and refusal evidence is now the ENGINE's line only:
  llama-swap's `http: proxy error` / argument errors never make a verdict.
- **A kill is not a restore.** `pkill -9` on a bench skips its `finally`: the temp entries
  stay deployed and the daily is not reloaded. Stop a run with SIGINT and wait, then check
  the box's yaml against the snapshot. And match the TARGET in the pattern
  (`bench.py --target pris`) - an interpreter-path pattern killed the other box's run
  (2026-09-25).
- **Update the masters' copy before trusting it.** A stale clone restored pris without its
  Swift entry (2026-09-25); pull it first, and the pre-run snapshot is what goes back regardless.
- **Restore what was there, not "the master".** A master path went
  away on 2026-09-22 and the restore would have thrown, leaving every temp entry deployed.
  The pre-run snapshot is the thing to put back.
- **The first pris run took five uploads** because each contract field was rediscovered
  from the dashboard source after the previous upload failed to render. The contract is
  the interface; read it first.
