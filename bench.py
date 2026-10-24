#!/usr/bin/env python3
"""LLMSpeedTest bench runner - the house instance of JR's reference suite.

Measures fleet boxes and posts each row to JunctionRelay OVER MCP as it is
measured (the only sanctioned channel - see ARCHITECTURE.md). Runs entirely
from one coordinating machine; once started it needs no agent and no model to be up, so the
model an opencode agent is riding can go down mid-run without consequence.

    python3 bench.py                      # interactive: pick from what the fleet serves
    python3 bench.py --all                # every model on every reachable box
    python3 bench.py --target pris        # one box, all its models
    python3 bench.py --target pris --models qwen3.8-27b-128k-q4   # exact set
    python3 bench.py ... --dry-run        # measure only, post nothing
    python3 bench.py ... --quick          # rungs 4k, 32k, 128k (--depths 4,32 for a custom set)

Start it DETACHED (nohup). Boxes are benched one at a time, in sequence.
Rows that fail to post stay in results/<target>_<stamp>.json - the outbox.
"""
import argparse
import json
import os
import random
import re
import shlex
import string
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime, timezone

# THE LAB'S BOXES live in targets.json beside this file (gitignored; JR_SPEED_TARGETS names
# another path). targets.example.json shows the shape: one entry per box, keyed by its role
# ("spark-a"), never its tin. Per field:
#   api           the OpenAI-compatible endpoint the bench measures
#   machine       the JR machine name the rows post under
#   ssh, peers    user@host of the box (and a pair's workers): GPU watts and the launch are read there
#   os            "linux" or "win"; engine "vllm" for a container engine with no llama-swap
#   yaml, model_root, master   llama-swap's config on the box, its model folder, and the
#                 master copy on this machine (--gap-fill and --entries; null on a container engine)
#   log_cmd       prints the engine's recent log over ssh. ⚠️ Tail DEEP and drop the suite's own
#                 polling (GET /running, /v1/models, metrics/activity): within seconds of a load
#                 failure the last 80 lines are polling noise and the cause line has scrolled away.
#                 A systemd-run llama-swap with logToStdout logs to the journal, not its log dir.
#   split_macros  per-model layer-split macros in the yaml, keyed by model dir
#   no_offload    true: this box never spills experts to RAM
#   preferred     the serving id to reload after a gap-fill, or null
#   hardware      the tin(s) the role runs on, " + " between a pair's ("Acer Veriton GN100 + ...")
def load_targets():
    path = os.environ.get("JR_SPEED_TARGETS") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "targets.json")
    try:
        targets = json.load(open(path))
    except FileNotFoundError:
        return {}
    for t in targets.values():
        if t.get("master"):
            t["master"] = os.path.expanduser(t["master"])
        t.setdefault("master", "")
    return targets


TARGETS = load_targets()
# Selector/alias ids are not models - benching one records a duplicate of
# whatever real model it resolves to, under a name the catalog doesn't have.
SKIP_IDS = {"brain"}
# Hugging Face orgs whose quant is THE quant of that format; anyone else is named on the row.
VENDOR_ORGS = {"nvidia", "qwen", "deepseek-ai", "zai-org"}
SOURCE = "LLMSpeedTest bench.py"

# ⛔ THE SUITE'S VERSION IS PART OF THE BENCHMARK CONTRACT. Speed numbers
# compare only within one version of the harness: change how a scenario is
# driven, what counts as the deep figure, or the salting, and the number means
# something different - exactly as a different engine build does. Bump this
# whenever a change could move a figure, and JR keeps the eras apart instead of
# averaging across a change nobody can see in the numbers.
#
#   1  the matrix era - four scenarios, salted prompts, telemetry-only speeds.
#   2  2026-09-24: `review` - the frozen Homelab docs corpus fed whole (~71.5K tokens).
#   3  2026-09-25: THE DEPTH LADDER, and nothing else. Scenarios are prompt depths
#      (`4k` `16k` `32k` `64k` `128k` `256k`, each rung the model's window allows)
#      plus `4k_x4` (four concurrent streams). Every rung is one cold request:
#      prompt_tok_s = prefill of the whole prompt, gen_tok_s = the next 128 tokens
#      at that depth (tg128), ttft_s = the server's prompt_ms. This is the shape
#      published numbers take (llama-bench -d, the vLLM/MLX recipes), so a cell
#      here reads against a cell out there. shallow/deep/review/parallel_x4 are
#      gone with their history (Jon: hardware moved too far for the old data to
#      matter). Prompts are the frozen corpus cut to the rung, cycled past 71.5K.
#   4  2026-10-18: a PUBLIC corpus, so any lab can run the suite and compare: three sections of
#      the Kubernetes concepts docs (corpus/README.md) replace the Homelab docs snapshot. Same
#      ladder and the same review ask; the prompt text differs, so v3 cells do not carry over.
#   5  2026-10-24: THE CONCURRENCY RUNG REBUILT. The `<depth>_xN` stage runs tg1024 at the
#      deepest rung the per-request window fits (32K down to 4K), so the decode windows
#      outlast the serialized prefills and the streams actually overlap; the row carries its
#      overlap factor and a WARNING note when they did not (each > total, or summed decode
#      time under 1.5x wall). v4's stage drove 4K prompts at tg128 - ~2 s of decode against a
#      ~9 s prefill queue - so on an engine that staggers streams its "each" was solo speed
#      measured inside a 4-stream test (the coding pair read "67 each / 30 total"). v4 xN
#      cells are not comparable to v5's; JR keeps the eras apart. The depth LADDER is
#      unchanged - only the concurrency rung moved.
SUITE = "LLMSpeedTest"
SUITE_NAME = "JR Speed Test"   # JR shows the suite by its own name (BENCHMARK-CONTRACT.md)
SUITE_VERSION = "5"
CORPUS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus", "kubernetes")
CORPUS_TOKENS = 51_400       # measured 2026-10-18 on GLM-5.3-Flash's tokenizer; calibrates chars/token
DEPTHS = (4, 16, 32, 64, 128, 256)   # K tokens; the ladder
QUICK_DEPTHS = (4, 32, 128)
TG_TOKENS = 128              # tg128: the industry decode length
TG_X = 1024                  # the concurrency rung's decode length (v5): the run must
                             # outlast the serialized prefills or the streams never
                             # decode together and "each" is solo speed (see the v5 era note)
LLB_DEPTHS = (4, 32, 128)    # --llama-bench: pp512 / tg128 at these K-token depths, the community's table
OPENCLAW_CFG = os.path.expanduser("~/.openclaw/openclaw.json")
REQUEST_TIMEOUT = 600  # a cold model load rides the first request


# ---------------------------------------------------------------- MCP client

class JRMcp:
    """Minimal MCP streamable-HTTP client. Credentials come from openclaw's
    lab-dev server entry so there is exactly one place they live."""

    def __init__(self):
        s = json.load(open(OPENCLAW_CFG))["mcp"]["servers"]["lab-dev"]
        self.url = s["url"]
        self.auth = s["headers"]["Authorization"]
        self.sid, self._id = None, 0
        r = self._post({"method": "initialize", "params": {
            "protocolVersion": "2025-03-26", "capabilities": {},
            "clientInfo": {"name": "LLMSpeedTest", "version": "2.0"}}})
        if "error" in r:
            raise RuntimeError(f"MCP initialize failed: {r['error']}")
        try:
            self._notify("notifications/initialized")
        except Exception:
            pass

    def _headers(self):
        h = {"Content-Type": "application/json",
             "Accept": "application/json, text/event-stream",
             "Authorization": self.auth}
        if self.sid:
            h["Mcp-Session-Id"] = self.sid
        return h

    def _post(self, payload):
        self._id += 1
        body = json.dumps({"jsonrpc": "2.0", "id": self._id, **payload}).encode()
        req = urllib.request.Request(self.url, data=body, headers=self._headers())
        with urllib.request.urlopen(req, timeout=60) as r:
            self.sid = r.headers.get("Mcp-Session-Id") or self.sid
            text, ctype = r.read().decode(), r.headers.get("Content-Type", "")
        if "text/event-stream" in ctype:
            for line in text.splitlines():
                if line.startswith("data:"):
                    msg = json.loads(line[5:].strip())
                    if msg.get("id") == self._id:
                        return msg
            raise RuntimeError("MCP: no response event in SSE body")
        return json.loads(text) if text.strip() else {}

    def _notify(self, method):
        body = json.dumps({"jsonrpc": "2.0", "method": method}).encode()
        urllib.request.urlopen(
            urllib.request.Request(self.url, data=body, headers=self._headers()),
            timeout=30).read()

    def call(self, tool, args, ride_out_s=150):
        # RIDE OUT A JR RESTART (2026-10-10): an update restarts the server, and a row posted in
        # that window was refused and lost - a run's whole 64k rung once. A connection failure
        # retries with backoff for up to ride_out_s before it counts as a failure.
        deadline, wait = time.time() + ride_out_s, 3
        while True:
            try:
                r = self._post({"method": "tools/call",
                                "params": {"name": tool, "arguments": args}})
                break
            except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
                if time.time() + wait > deadline:
                    raise
                print(f"  .. JR unreachable ({e}); retrying in {wait}s")
                time.sleep(wait)
                wait = min(wait * 2, 30)
        if "error" in r:
            raise RuntimeError(f"MCP {tool}: {r['error']}")
        result = r.get("result", {})
        text = "".join(c.get("text", "") for c in result.get("content", []))
        if result.get("isError"):
            raise RuntimeError(f"MCP {tool}: {text}")
        return text

    def model_ids(self):
        return {m.group(2): int(m.group(1))
                for m in re.finditer(r"\[(\d+)\]\s+(\S+)", self.call("models_query", {}))}


# A serving id names a CONFIGURATION; the catalog holds MODELS WE HAVE
# DOWNLOADED. Quant, context window and MTP are ways of RUNNING a model, so they
# are stripped to find the catalog entry:
#
#     qwen3.8-27b-256k-q4-mtp  ->  qwen3.8-27b
#
# ⚠️ The quant strip is the part that is easy to leave out, and it fails silently
# when you do - the run measures fine and the results are simply never posted.
# The catalog collapsed quant-named entries into their models on 2026-08-27, so
# an id resolving to 'qwen3.8-27b-q4' matches nothing any more. The quant suffix
# must start with q/iq, which is what keeps it off names like 'lfm2.5-8b-a1b'
# and 'qwen3-embedding-0.6b'.
def catalog_name(serving_id):
    s = serving_id.removesuffix("-mtp")
    s = re.sub(r"-\d+k(?=-|$)", "", s)
    # ⚠️ NAMED FORMATS TOO, not just -q4/-iq4. NVFP4 is a quantisation of the
    # same checkpoint and lives in the catalog as a QUANT of it, so
    # 'qwen3.8-27b-128k-nvfp4-high' must resolve to 'qwen3.8-27b'. Without this
    # it resolved to 'qwen3.8-27b-nvfp4-high', matched nothing, and the rows
    # rendered "not in catalog" against a model that is very much in it
    # (2026-08-28).
    s = re.sub(r"-(?:nvfp4|fp8|bf16|f16)(?:[-_][a-z0-9]+)*$", "", s, flags=re.I)
    # TensorFold served ids carry the format too (GLM-5.3-Flash-EXL3, 2026-10-03): EXL3 / MLX / AWQ /
    # GPTQ are quants of the catalog's checkpoint, and the catalog spells names in lower case.
    s = re.sub(r"-(?:exl3|mlx|awq|gptq)(?:[-_][a-z0-9.]+)*$", "", s, flags=re.I).lower()
    # A slots token after the quant ('-q5-4s') left the quant in the name and
    # posted a run as 'qwen3.8-27b-q5-4s', a model in no catalog (2026-09-24).
    s = re.sub(r"-\d+s(?=-|$)", "", s, flags=re.I)
    s = re.sub(r"-ub\d+$", "", s, flags=re.I)
    # A placement tag on a re-bench id ('-ram37': the same name as a served
    # entry, a different placement) is not a model either - it minted a phantom
    # 'qwen3.8-flash-next-reap256-q3-ram37' in the ledger (2026-09-25).
    s = re.sub(r"-ram\d+(?=-|$)", "", s, flags=re.I)
    # A batch tag ('-ub512', the ub sweep of 2026-09-25) is tuning, not a model.
    s = re.sub(r"-ub\d+(?=-|$)", "", s, flags=re.I)
    return re.sub(r"-(?:iq|q)\d[a-z0-9_]*$", "", s, flags=re.I)


def resolve_model_id(ids, serving_id):
    """Catalog id for a serving id, trying progressively less specific names."""
    return (ids.get(serving_id)
            or ids.get(serving_id.removesuffix("-mtp"))
            or ids.get(re.sub(r"-\d+k(?=-|$)", "", serving_id.removesuffix("-mtp")))
            or ids.get(catalog_name(serving_id)))


# ------------------------------------------------------------- measurement

def http_json(url, payload=None, timeout=REQUEST_TIMEOUT):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def salt(n=10):
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=n))


_corpus_cache = None


def corpus_files():
    """The frozen corpus (corpus/README.md) as (relpath, text), sorted, loaded once."""
    global _corpus_cache
    if _corpus_cache is None:
        out = []
        for root, _, files in sorted(os.walk(CORPUS_DIR)):
            for name in sorted(files):
                path = os.path.join(root, name)
                try:
                    out.append((os.path.relpath(path, CORPUS_DIR).replace(os.sep, "/"),
                                open(path, encoding="utf-8", errors="replace").read()))
                except OSError:
                    continue
        if not out:
            raise RuntimeError(f"corpus missing or empty: {CORPUS_DIR}")
        _corpus_cache = out
    return _corpus_cache


def warmup_prompt():
    return f"(session {salt()}) Hello! In one or two sentences, how are you today?"


# A rung's prompt: the corpus cut to ~k thousand tokens at a file boundary,
# cycled with a pass marker past its 51.4K, then the review ask. The salt goes
# FIRST and the request carries cache_prompt=false, so no run prefills from
# cache (llama-server's --cache-reuse splices matching chunks otherwise).
# Depth is a target; the row's note carries the prompt_tokens the server counted.
def ladder_prompt(k_tokens, stream=None):
    files = corpus_files()
    cpt = sum(len(t) for _, t in files) / CORPUS_TOKENS
    budget = int(k_tokens * 1000 * cpt) - 400
    parts, total, i = [], 0, 0
    while total < budget:
        rel, text = files[i % len(files)]
        chunk = f"\n\n===== {rel} (pass {i // len(files) + 1}) =====\n{text}"
        if total + len(chunk) > budget:
            chunk = chunk[:budget - total]
        parts.append(chunk)
        total += len(chunk)
        i += 1
    head = f"(review session {salt()}" + (f" stream {stream}" if stream is not None else "") + ") "
    return (head + "You are reviewing the documentation repository below. Read every "
            "file. Then report, with the file path for each finding: contradictions between "
            "pages, stale facts, procedures that cannot be followed as written, and the three "
            "changes that would most improve the repo. Be specific and concise."
            + "".join(parts))


def latest_activity(api, model):
    try:
        records = http_json(f"{api}/api/metrics/activity", timeout=15)
    except Exception:
        return None
    for rec in reversed(records):
        if rec.get("model") == model and rec.get("prompt_tokens", 0) > 0:
            return rec
    return None


def run_one(api, model, prompt, max_tokens, cache_prompt=True, timeout=REQUEST_TIMEOUT):
    """One completion. Speeds come from engine telemetry - the response's own
    timings block when the build provides it, else the activity log; wall
    clock is never reported as tok/s."""
    payload = {"model": model,
               "messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "stream": False}
    if not cache_prompt:
        payload["cache_prompt"] = False  # llama-server honours it on /v1 too
    resp = http_json(f"{api}/v1/chat/completions", payload, timeout=timeout)
    usage = resp.get("usage", {})
    out = {"output_tokens": usage.get("completion_tokens", 0),
           "prompt_tokens": usage.get("prompt_tokens", 0),
           "gen_tps": None, "prompt_tps": None, "ttft_s": None, "cache_n": None, "accept": None}
    timings = resp.get("timings") or {}
    if timings.get("predicted_per_second"):
        out["gen_tps"] = timings["predicted_per_second"]
        out["prompt_tps"] = timings.get("prompt_per_second")
        if timings.get("prompt_ms"):
            out["ttft_s"] = timings["prompt_ms"] / 1000
        out["cache_n"] = timings.get("cache_n")
        if timings.get("draft_n"):
            out["accept"] = timings.get("draft_n_accepted", 0) / timings["draft_n"]
    else:
        rec = latest_activity(api, model) or {}
        out["gen_tps"] = rec.get("tokens_per_second")
        out["prompt_tps"] = rec.get("prompt_per_second")
    return out


# ------------------------------------------------------------------- vLLM
# vLLM answers with usage only, so the engine's telemetry is its Prometheus
# /metrics: prefill time (scheduled -> first token), decode time (first -> last
# token), token and speculation counters. A request's figures are the DELTA
# across it, which is only its own when nothing else finished meanwhile - the
# box must be idle, and a delta that counts other requests is thrown away.

def engine_of(api, default="vllm"):
    """Which engine answers at `api`, from its own /metrics: TensorFold publishes tensorfold:*
    series, vLLM vllm:*. The target table cannot know - a Spark's engine changes with its recipe."""
    try:
        with urllib.request.urlopen(f"{api}/metrics", timeout=15) as r:
            text = r.read().decode(errors="replace")
    except Exception:
        return default
    if "\ntensorfold:" in "\n" + text:
        return "tensorfold"
    if "\nvllm:" in "\n" + text:
        return "vllm"
    return default


def vllm_metrics(api, model):
    """name -> value for one served model, summed over its label sets. vLLM series carry a
    model_name label and are filtered on it; TensorFold serves one model and labels its series
    by pool/stream only, so every tensorfold:* series counts (tensorfold_health:* as health_*).
    Two TensorFold builds publish different names: the 0.6.0 GLM recipe has health_prefill_
    seconds_total / health_decode_seconds_total, the 0.6.1 Flash-Next recipe has
    time_to_first_token_seconds_sum and request_latency_seconds_sum; tf_deltas reads both."""
    with urllib.request.urlopen(f"{api}/metrics", timeout=15) as r:
        text = r.read().decode(errors="replace")
    out = {}
    for ln in text.splitlines():
        m = re.match(r"vllm:(\w+)\{([^}]*)\}\s+(\S+)", ln)
        if m and f'model_name="{model}"' in m.group(2):
            out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(3))
            continue
        m = re.match(r"tensorfold(_health)?:(\w+)(?:\{[^}]*\})?\s+(\S+)", ln)
        if m and "_bucket" not in m.group(2):
            key = ("health_" if m.group(1) else "") + m.group(2)
            try:
                out[key] = out.get(key, 0.0) + float(m.group(3))
            except ValueError:
                pass
    return out


def tf_deltas(d):
    """TensorFold's (finished requests, prefill seconds, decode seconds, drafted, accepted)
    from a metrics delta, whichever of its two metric sets the build publishes."""
    n = d.get("health_requests_total") or d.get("request_latency_seconds_count") or d.get("e2e_request_latency_seconds_count") or 0
    if d.get("health_prefill_seconds_total") is not None and "health_prefill_seconds_total" in d:
        prefill, decode = d.get("health_prefill_seconds_total", 0), d.get("health_decode_seconds_total", 0)
    elif (d.get("request_prefill_time_seconds_sum") is not None
          or d.get("request_decode_time_seconds_sum") is not None
          or d.get("request_decode_seconds_sum") is not None):
        # The build states its own phase seconds per request (exl3-tp2 dsv41 does). Prefer
        # those over inferring: request_latency - ttft is the first-to-last-token SPAN, and
        # a stream descheduled while another's prefill ran pays that gap in the span.
        prefill = d.get("request_prefill_time_seconds_sum", d.get("request_prefill_seconds_sum", 0.0))
        decode = d.get("request_decode_time_seconds_sum", d.get("request_decode_seconds_sum", 0.0))
        if not decode:
            decode = max(0.0, d.get("request_latency_seconds_sum", 0.0) - prefill)
    else:
        ttft = d.get("time_to_first_token_seconds_sum", 0)
        prefill, decode = ttft, max(0.0, d.get("request_latency_seconds_sum", 0) - ttft)
    drafted = d.get("mtp_drafted_total") or d.get("spec_decode_num_draft_tokens_total") or 0
    accepted = d.get("mtp_accepted_total") or d.get("spec_decode_num_accepted_tokens_total") or 0
    return n, prefill, decode, drafted, accepted


def vllm_wait_idle(api, model, timeout=900):
    end = time.time() + timeout
    while time.time() < end:
        m = vllm_metrics(api, model)
        if not (m.get("num_requests_running") or m.get("requests_running")) and \
           not (m.get("num_requests_waiting") or m.get("requests_waiting")):
            return
        time.sleep(5)
    print("  ⚠ vLLM still busy with someone else's requests - proceeding, the delta check decides")


def run_vllm(api, model, make_prompts, max_tokens, timeout=REQUEST_TIMEOUT, tries=3):
    """len(make_prompts()) concurrent completions against vLLM; one result dict
    per stream, in run_one's shape plus the batch's `wall`. Speeds are the
    /metrics deltas across the batch (per-stream rates: summed tokens over
    summed engine time). make_prompts is called per attempt - a retried prompt
    would prefill from the prefix cache. Raises when a stream errors, or when
    every attempt saw another client's request finish inside the window."""
    for attempt in range(tries):
        vllm_wait_idle(api, model)
        prompts = make_prompts()
        n = len(prompts)
        resps = [None] * n

        def work(i):
            try:
                resps[i] = http_json(f"{api}/v1/chat/completions", {
                    "model": model, "messages": [{"role": "user", "content": prompts[i]}],
                    "max_tokens": max_tokens, "stream": False}, timeout=timeout)
            except Exception as e:
                resps[i] = e

        before = vllm_metrics(api, model)
        tf = "prompt_tokens_total" in before and "request_success_total" not in before   # TensorFold, not vLLM
        start = time.time()
        threads = [threading.Thread(target=work, args=(i,)) for i in range(n)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        wall = time.time() - start
        for r in resps:
            if isinstance(r, Exception):
                raise r
        # The histograms are written as the request finishes; give them a moment.
        for _ in range(20):
            after = vllm_metrics(api, model)
            d = {k: after.get(k, 0.0) - before.get(k, 0.0) for k in after}
            finished = tf_deltas(d)[0] if tf else d.get("request_success_total", 0)
            if finished >= n:
                break
            time.sleep(0.5)
        if tf:
            finished, prefill, decode, drafted, accepted = tf_deltas(d)
            clean = round(finished) == n
        else:
            finished = d.get("request_success_total", 0)
            clean = round(finished) == n and round(d.get("request_prefill_time_seconds_count", 0)) == n
        if not clean:
            print(f"  another client's request landed in the window "
                  f"({finished:.0f} finished, {n} ours) - "
                  + ("retrying" if attempt + 1 < tries else "giving up"))
            continue
        usage = [r.get("usage", {}) for r in resps]
        ptoks = sum(u.get("prompt_tokens", 0) for u in usage)
        gtoks = sum(u.get("completion_tokens", 0) for u in usage)
        cached = sum((u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0 for u in usage)
        if not tf:
            prefill = d.get("request_prefill_time_seconds_sum", 0)
            decode = d.get("request_decode_time_seconds_sum", 0)
            drafted = d.get("spec_decode_num_draft_tokens_total", 0)
            accepted = d.get("spec_decode_num_accepted_tokens_total", 0)
        batch = {
            # Prefill rate over the tokens actually computed: the template's
            # fixed system block ahead of the salt is a prefix-cache hit.
            "prompt_tps": (ptoks - cached) / prefill if prefill else None,
            # Decode time runs first token -> last, so it covers gtoks - n tokens.
            "gen_tps": (gtoks - n) / decode if decode and gtoks > n else None,
            "ttft_s": prefill / n if prefill else None,
            "accept": accepted / drafted if drafted else None,
            "wall": wall,
        }
        return [{"output_tokens": u.get("completion_tokens", 0),
                 "prompt_tokens": u.get("prompt_tokens", 0),
                 "cache_n": (u.get("prompt_tokens_details") or {}).get("cached_tokens") or None,
                 **batch} for u in usage]
    raise RuntimeError("vLLM was never idle for a whole request - no clean measurement")


def unwrap_shell(argv):
    """A compose launch is often ["bash", "-lc", "A=1; ...; exec vllm serve ..."]: split the string,
    resolve the $VAR / ${VAR} its own assignments set, and return the serve argv (2026-10-10, DSpark)."""
    if not (argv and len(argv) >= 3 and argv[0].split("/")[-1] in ("sh", "bash") and argv[-2].startswith("-") and "c" in argv[-2]):
        return argv
    try:
        toks = shlex.split(argv[-1].replace(";", " ; "))
    except ValueError:
        return argv
    vals = {}
    for tok in toks:
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$", tok)
        if m:
            vals[m.group(1)] = m.group(2)
    sub = lambda tok: re.sub(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)(?::-[^}]*)?\}?", lambda m: vals.get(m.group(1), m.group(0)), tok)
    for i, tok in enumerate(toks):
        if tok.endswith("vllm") and i + 1 < len(toks) and toks[i + 1] == "serve":
            out = []
            for x in toks[i:]:
                if x == ";":
                    break
                out.append(sub(x))
            return out
    return argv


def serve_argv_in(t, cid):
    """The argv of the `vllm serve` process running inside container `cid`, from /proc. A launch
    wrapped in a script (MiaAI's DeepSeek-V4.1 EXL3 runs /start.sh) builds its flags at run time,
    so Config.Cmd says only '/start.sh' - and the rows posted no MTP and the wrong slots (2026-10-10)."""
    code, out = _ssh(t, f"docker exec {cid[:12]} sh -c 'for p in /proc/[0-9]*/cmdline; do tr \"\\\\0\" \"\\\\t\" < $p; echo; done' 2>/dev/null")
    for ln in out.splitlines():
        toks = [x for x in ln.split("\t") if x]
        if "serve" in toks:
            i = toks.index("serve")
            if i > 0 and toks[i - 1].endswith("vllm"):
                return ["vllm"] + toks[i:]
    return None


def vllm_launch(t, model):
    """(argv, env, hf_cache) of the container serving `model`, from docker
    inspect over ssh - the launch is the only source of the setup flags."""
    _, out = _ssh(t, "docker inspect $(docker ps -q) --format "
                     "'{{json .Id}}\t{{json .Config.Cmd}}\t{{json .Config.Env}}\t{{json .Mounts}}'")
    for ln in out.splitlines():
        try:
            cid, argv, env, mounts = (json.loads(x) for x in ln.split("\t"))
        except ValueError:
            continue
        argv = unwrap_shell(argv)
        if "serve" not in argv:
            argv = serve_argv_in(t, cid) or argv   # a script launch (/start.sh) - read the live serve process
        if argv and (("--served-model-name" in argv and argv[argv.index("--served-model-name") + 1] == model)
                     or (len(argv) > 1 and os.path.basename(argv[0]).startswith("tensorfold") and argv[1] == "serve"
                         and "--name" in argv[:-1] and argv[argv.index("--name") + 1] == model)):   # tensorfold or tensorfold-native (Zig)
            hf = next((m["Source"] for m in mounts if m["Destination"].endswith(".cache/huggingface") or m["Destination"] == "/hf"), None)   # /hf: the DeepSeek-V4.1 recipe
            return argv, env or [], hf
    return [], [], None


def source_repo(t, path, mounts=None):
    """The Hugging Face repo of the weights a server loaded - JR links the row by it (JR 0.9.0.272,
    sourceRepo). An HF cache path names it (models--owner--repo); any other folder carries the
    archive downloader's .hf-repo file beside the weights. A container path is mapped to the
    host through its mounts first. None when neither says - the row then links by name."""
    if not path:
        return None
    m = re.search(r"models--([^/\\]+?)--([^/\\]+)", path)
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    if not path.startswith(("/", ".")) and ":" not in path and re.fullmatch(r"[\w.-]+/[\w.-]+", path):
        return path                                   # already a repo id (a TensorFold launch names one)
    host = path
    for mt in mounts or []:
        dst = mt.get("Destination", "")
        if dst and (path == dst or path.startswith(dst.rstrip("/") + "/")):
            host = mt.get("Source", "") + path[len(dst.rstrip("/")):]
            break
    folder = host if not re.search(r"\.(gguf|safetensors)$", host, re.I) else re.split(r"[/\\][^/\\]+$", host)[0]
    if t.get("os") == "win":
        code, out = _ssh(t, f'type "{folder}\\.hf-repo"')
    else:
        code, out = _ssh(t, f"cat '{folder}/.hf-repo' 2>/dev/null")
    repo = out.strip().splitlines()[0].strip() if code == 0 and out.strip() else ""
    return repo if re.fullmatch(r"[\w.-]+/[\w.-]+", repo) else None


def quant_of(root):
    """The quant a checkpoint repo name declares, with its publisher unless a vendor org:
    'NVFP4-Mia', 'EXL3-4bpw-Mia', 'MLX-4bit-Vontra', 'FP8' (nvidia). The archive folders and
    the HUD use the same spelling (ARCHITECTURE.md, 'a third party's quant carries its publisher')."""
    name = root.split("/")[-1]
    q = re.search(r"(EXL3-?(?:EAQ-v[\d.]+-)?\d(?:\.\d+)?-?bpw|MLX-\d+bit|NVFP4(?:-[A-Z]+)?|FP8|AWQ|BF16)", name, re.I)
    if not q:
        return None
    # EXL3-4bpw, and Mia's EAQ builds (EXL3-EAQ-v1-2.9-bpw) in the same spelling: EXL3-2.9bpw
    quant = re.sub(r"^EXL3-?(?:EAQ-v[\d.]+-)?(\d(?:\.\d+)?)-?bpw", r"EXL3-\1bpw", q.group(1), flags=re.I)
    org = root.split("/")[0] if "/" in root else ""
    if org and org.lower() not in VENDOR_ORGS:
        quant += "-" + re.split(r"[-_]", org)[0]
    return quant


def config_tensorfold(t, model):
    """config_json for a TensorFold container (Mia's Spark recipes): the launch argv is
    `tensorfold serve <repo> --name <model> --parallel N --context C --kv-dtype D ...`; the
    engine version is the image tag; drafting is on unless --no-drafts; thinking is a server
    default here (--thinking / --no-thinking), so it is recorded as the setup fact it is."""
    cfg = {"llama_swap_name": model, "suite": SUITE, "suite_name": SUITE_NAME, "suite_version": SUITE_VERSION}
    argv, env, hf = vllm_launch(t, model)
    flag = lambda f: argv[argv.index(f) + 1] if f in argv[:-1] else None
    code, out = _ssh(t, "docker ps --format '{{.Names}}\t{{.Image}}' | grep -E -- '-tf$|tensorfold|^tf-' | head -1")
    image = out.strip().split("\t")[-1] if code == 0 and out.strip() else ""
    ver = re.search(r":(v?\d[\w.-]*)", image) or re.search(r":([\w.-]+)$", image)   # :v0.6.1, :0.6.2-pr141, Zig :zig-db28187
    cfg["engine"] = "TensorFold " + (ver.group(1) if ver else "?") + (f" ({image})" if image else "")
    # The runtime as data (JR 0.9.0.266): the pages show name + version and never parse engine.
    cfg["engine_name"], cfg["engine_version"] = "TensorFold", (ver.group(1).lstrip("v") if ver else None)
    _, lbl = _ssh(t, f"docker image inspect {image} --format '{{{{index .Config.Labels \"tf.patches\"}}}}'") if image else (1, "")
    if lbl.strip():
        cfg["engine"] += f" patches {lbl.strip()}"
    if flag("--context"):
        cfg["ctx"] = int(flag("--context"))
    if flag("--parallel"):
        cfg["parallel"] = int(flag("--parallel"))
    # KV: --kv-dtype on the Flash-Next recipe; the GLM pair recipe passes it as TF_GLM_KV in the
    # environment (fp8) and the server default is bf16 when neither says.
    kv_env = next((e.split("=", 1)[1] for e in env if e.startswith("TF_GLM_KV=")), None)
    if flag("--kv-dtype") or kv_env:
        cfg["cache_k"] = cfg["cache_v"] = flag("--kv-dtype") or kv_env
    elif "tensorfold-dsv41" in image:   # DeepSeek-V4.1's engine takes no KV dtype: the model's own layout
        cfg["cache_k"] = cfg["cache_v"] = "native"
    # DeepSeek-V4.1's TensorFold recipe reads its engram tables from a file on the NVMe by design (TF_DS_ENGRAM):
    # the weights are all in VRAM, so its disk reads are not paging (2026-10-23: its rows grouped as "paged from SSD")
    if any(e.startswith("TF_DS_ENGRAM=") for e in env):
        cfg["engram_offload"] = "ssd"
    if flag("--tp"):
        cfg["tensor_parallel"] = int(flag("--tp"))
    # Drafting: an MTP head (Flash-Next) or a DFlash2 drafter checkpoint (--drafter, the GLM pair);
    # both verify exactly and both are "speculative decoding on" to the ledger's mtp flag.
    if flag("--drafter"):
        cfg["spec_type"] = "dflash2"
        cfg["drafter"] = re.sub(r".*models--([^/]+)--([^/]+)/.*", r"\1/\2", flag("--drafter"))
    elif "--no-drafts" not in argv:
        cfg["spec_type"] = "mtp"
        if flag("--mtp-drafts"):
            cfg["spec_tokens"] = int(flag("--mtp-drafts"))
    # Thinking as the SERVER DEFAULT, what a client that sends nothing gets: off; on (Qwen's
    # boolean template); max (GLM's effort-level template: TensorFold applies its maximum when a
    # request sends no reasoning_effort - measured 2026-10-02, 1,050 thinking tokens on a riddle).
    root = argv[2] if len(argv) > 2 else ""
    m = re.search(r"models--([^/]+)--([^/]+)(?:/|$)", root)
    if m:
        root = f"{m.group(1)}/{m.group(2)}"          # the cache path names the repo
    elif root.startswith("/hf/") and hf:
        # DeepSeek-V4.1's recipe serves a folder of links into the HF cache (/hf/tensorfold-dsv41/<rev>/model):
        # one link names the repo (2026-10-23: its rows had posted with no quant)
        _, ln = _ssh(t, f"readlink -f '{hf}{root[3:]}/config.json'")
        m = re.search(r"models--([^/]+)--([^/]+)/", ln or "")
        if m:
            root = f"{m.group(1)}/{m.group(2)}"
    if "--no-thinking" in argv:
        cfg["thinking_default"] = "off"
    elif "--thinking" in argv:
        cfg["thinking_default"] = "max" if re.search(r"glm", root, re.I) else "on"
        # The server's own effort level, when the launch sets one: sfxnz's pair adds --reasoning-effort low
        # (pair.env EXTRA_ARGS), and its rows read "on" (2026-10-10). A request that sends nothing gets this.
        if flag("--reasoning-effort"):
            cfg["thinking_default"] = flag("--reasoning-effort")
    if "--ple-on-ssd" in argv:
        cfg["ple_offload"] = "ssd"
    q = quant_of(root)
    if q:
        cfg["quant"] = q
    cfg["source_repo"] = source_repo(t, root if re.fullmatch(r"[\w.-]+/[\w.-]+", root) else (argv[2] if len(argv) > 2 else ""))
    if hf and root:
        code, out = _ssh(t, f"du -sbL '{hf}/hub/models--{root.replace('/', '--')}/snapshots' | cut -f1")
        if code == 0 and out.strip().isdigit():
            cfg["weights_gb"] = round(int(out.strip()) / 1e9, 2)
    return cfg


def thinking_probe(api, model):
    """The server's thinking default, MEASURED when the launch does not state it: one short request
    that sends no thinking setting, 'on' if reasoning comes back, else 'off'. A failed probe is None
    (the row says nothing) - never a guess."""
    try:
        r = http_json(f"{api}/v1/chat/completions",
                      {"model": model, "max_tokens": 512,
                       "messages": [{"role": "user", "content": f"(probe {salt()}) What is 17 x 23? Reply with the number."}]},
                      timeout=600)
        msg = r["choices"][0]["message"]
    except Exception:
        return None
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    return "on" if reasoning.strip() or "<think>" in (msg.get("content") or "") else "off"


def vision_probe(api, model):
    """Whether the server takes an image, MEASURED: one request carrying a 1x1 PNG. Answered = True;
    refused with a client error (vLLM: 'not a multimodal model') = False; anything else = None."""
    png = ("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4"
           "z8DwHwAFAAH/iZk9HQAAAABJRU5ErkJggg==")
    body = {"model": model, "max_tokens": 8,
            "messages": [{"role": "user", "content": [{"type": "text", "text": "Colour?"},
                                                      {"type": "image_url", "image_url": {"url": png}}]}]}
    try:
        http_json(f"{api}/v1/chat/completions", body, timeout=300)
        return True
    except urllib.error.HTTPError as e:
        return False if 400 <= e.code < 500 else None
    except Exception:
        return None


def config_vllm(t, model):
    """config_json for a vLLM target: every fact from the container's launch
    argv or the server itself. Absent is honest, guessed is not."""
    if engine_of(t["api"]) == "tensorfold":
        return config_tensorfold(t, model)
    cfg = {"llama_swap_name": model, "suite": SUITE, "suite_name": SUITE_NAME, "suite_version": SUITE_VERSION}
    argv, env, hf = vllm_launch(t, model)
    flag = lambda f: argv[argv.index(f) + 1] if f in argv[:-1] else None
    try:
        full = http_json(f"{t['api']}/version", timeout=15)["version"]
        cfg["engine"] = "vLLM " + full
        cfg["engine_version"] = full.split("+")[0]
        cfg["engine_name"] = "vLLM"
    except Exception:
        pass
    try:
        info = next(m for m in http_json(f"{t['api']}/v1/models", timeout=15)["data"] if m["id"] == model)
    except Exception:
        info = {}
    if flag("--max-model-len") or info.get("max_model_len"):
        cfg["ctx"] = int(flag("--max-model-len") or info["max_model_len"])
    if flag("--max-num-seqs"):
        cfg["parallel"] = int(flag("--max-num-seqs"))
    if flag("--kv-cache-dtype") and flag("--kv-cache-dtype") != "auto":
        cfg["cache_k"] = cfg["cache_v"] = flag("--kv-cache-dtype")
    else:
        # 'auto' is the ENGINE's choice, not f16: DeepSeek-V4.1 picks fp8_ds_mla and says so in its log
        # ("Using DeepSeek's fp8_ds_mla KV cache format."). Rows that read auto as f16 were wrong (2026-10-10).
        _, kvlog = _ssh(t, "docker logs $(docker ps -q | head -1) 2>&1 | grep -m1 -oE \"Using [^ ]+ [^ ]+ KV cache format\"")
        m = re.search(r"Using \S+ (\S+) KV cache format", kvlog)
        if m:
            cfg["cache_k"] = cfg["cache_v"] = m.group(1)
    # VISION from the launch: --limit-mm-per-prompt says how many images a request may carry (V4.1-Flash
    # EXL3: {"image":100}); 0 images or --language-model-only means text only. Neither flag = unknown,
    # posted as nothing. vLLM rows never carried vision before (2026-10-10).
    if "--language-model-only" in argv:
        cfg["vision"] = False
    elif flag("--limit-mm-per-prompt"):
        try:
            cfg["vision"] = int(json.loads(flag("--limit-mm-per-prompt")).get("image", 0)) > 0
        except (ValueError, AttributeError, TypeError):
            pass
    if "vision" not in cfg:
        cfg["vision"] = vision_probe(t["api"], model)   # neither flag (DeepSeek-V4-Flash 0731's DSpark launch)
    if flag("--tensor-parallel-size"):
        cfg["tensor_parallel"] = int(flag("--tensor-parallel-size"))
    if flag("--gpu-memory-utilization"):
        cfg["gpu_memory_utilization"] = float(flag("--gpu-memory-utilization"))
    try:
        spec = json.loads(flag("--speculative-config") or "{}")
    except ValueError:
        spec = {}
    # The server's own thinking default, when its launch sets one (DSpark: {"thinking":false}).
    try:
        kw = json.loads(flag("--default-chat-template-kwargs") or "{}")
    except ValueError:
        kw = {}
    for k in ("thinking", "enable_thinking"):
        if isinstance(kw.get(k), bool):
            cfg["thinking_default"] = "on" if kw[k] else "off"
    if "thinking_default" not in cfg:
        th = thinking_probe(t["api"], model)
        if th:
            cfg["thinking_default"] = th
    if spec.get("method"):
        cfg["spec_type"] = spec["method"]
        if spec.get("num_speculative_tokens"):
            cfg["spec_tokens"] = spec["num_speculative_tokens"]
    # The single-node recipes memory-map the per-layer embedding table from the
    # SSD; whether they did is the placement fact on a unified-memory box.
    ple = next((e.split("=", 1)[1] for e in env if e.startswith("PLE_OFFLOAD=")), None)
    if ple is not None:
        cfg["ple_offload"] = ple
    # A file-backed engram table (DeepSeek-V4.1 EXL3: hf_overrides engram_table_dir) lives on the SSD by design.
    try:
        if json.loads(flag("--hf-overrides") or "{}").get("engram_table_dir"):
            cfg["engram_offload"] = "ssd"
    except (ValueError, AttributeError):
        pass
    root = info.get("root") or (argv[0] if argv else "")
    _, mj = _ssh(t, "docker inspect $(docker ps -q) --format '{{json .Mounts}}'")
    mounts = [x for ln in mj.splitlines() if ln.strip().startswith("[") for x in json.loads(ln)]
    cfg["source_repo"] = source_repo(t, root, mounts)
    q = re.search(r"(NVFP4(?:-[A-Z]+)?|FP8|AWQ|BF16)", root.split("/")[-1], re.I)
    if q:
        cfg["quant"] = q.group(1)
        # A third party's quantisation of the same checkpoint is a different file under the
        # same repo name: Mia-AiLab's NVFP4 is 105.9 GB, nvidia's 132.7 GB. The publisher goes
        # on the quant ('NVFP4-Mia', as the archive folder is named) or the two share a row.
        org = root.split("/")[0] if "/" in root else ""
        if org and org.lower() not in VENDOR_ORGS:
            cfg["quant"] += "-" + re.split(r"[-_]", org)[0]
    if not cfg.get("quant") and root.startswith("/"):
        # A local folder name may say nothing ("DeepSeek-V4-Flash-0731-native"): the checkpoint's own
        # config.json does. A vendor's checkpoint carries no publisher suffix (2026-10-10).
        _, out = _ssh(t, f"for c in $(docker ps -q); do docker exec $c cat '{root}/config.json' 2>/dev/null && break; done")
        try:
            qc = json.loads(out).get("quantization_config") or {}
        except ValueError:
            qc = {}
        method = str(qc.get("quant_method") or "").lower()
        names = {"fp8": "FP8", "mxfp4": "MXFP4", "nvfp4": "NVFP4", "awq": "AWQ", "gptq": "GPTQ"}
        if method in names:
            cfg["quant"] = names[method]
    if hf and root:
        code, out = _ssh(t, f"du -sbL '{hf}/hub/models--{root.replace('/', '--')}/snapshots' | cut -f1")
        if code == 0 and out.strip().isdigit():
            cfg["weights_gb"] = round(int(out.strip()) / 1e9, 2)
    return cfg


def vllm_fit(api, model, prompt_for, target, ctx):
    """The rung's target (K tokens), pulled under the window when the prompt
    would overflow it: vLLM counts exactly (/tokenize) and refuses the request."""
    if not ctx or target * 1000 * 1.1 + TG_TOKENS < ctx:
        return target
    try:
        n = http_json(f"{api}/tokenize", {"model": model, "messages": [
            {"role": "user", "content": prompt_for(target)}]}, timeout=300)["count"]
    except Exception:
        return target
    room = ctx - TG_TOKENS - 1024
    return target * room / n if n > room else target


def running_cmd(api, model):
    """The launch command llama-swap actually spawned for this model - the
    ONLY remote source of truth for spec/KV/MoE flags. /props does not report
    them (verified on b10360 and b10639: --spec-type draft-mtp still shows
    speculative.types 'none' there)."""
    try:
        for rec in http_json(f"{api}/running", timeout=15).get("running", []):
            if rec.get("model") == model:
                return rec.get("cmd", "")
    except Exception:
        pass
    return ""


def placement_from_cmd(cmd):
    """WHERE THE WEIGHTS SIT, from a launch command: expert layers in RAM (the
    --n-cpu-moe range AND every -ot ...=CPU expert pin, counted), load mode, the
    tensor overrides, the ubatch and the GPU split.

    One function for BOTH a measured row (config_json, from /running) and a
    refusal (cfg_from_yaml, from the deployed entry). It used to live only in
    config_json, so a refused load posted a verdict with no placement at all and
    JR filed it under all-in-VRAM whatever the entry pinned (2026-09-25: the
    REAP + MTP round B refusal had a layer in RAM and read as all on the cards).
    """
    cfg = {}
    m = re.search(r"--n-cpu-moe\s+(\d+)", cmd)
    if m:
        cfg["n_cpu_moe"] = int(m.group(1))
    # ⚠️ THE FLAGS THAT DECIDE WHERE WEIGHTS LIVE. Without these a reader of the
    # ledger sees 20 tok/s next to 65 and cannot tell that one was mmapped and
    # paging off SSD while the other was resident (2026-08-28).
    #
    # --load-mode none disables mmap and forces a resident load. It is REQUIRED
    # with --n-cpu-moe (house measurement 2026-08-15: mmap plus CPU tensor
    # overrides collapsed prefill to 66 tok/s), so its ABSENCE alongside
    # n_cpu_moe is itself the explanation for a bad number.
    m = re.search(r"--load-mode\s+(\S+)", cmd)
    cfg["load_mode"] = m.group(1) if m else "mmap"
    # Explicit tensor placement, e.g. -ot ple_ngram_embd=CPU.
    ots = re.findall(r"-ot\s+(\S+)", cmd)
    if ots:
        cfg["tensor_overrides"] = ots
    # ⛔ EXPERT LAYERS PINNED TO CPU BY NAME ARE OFFLOAD TOO. The regime was read
    # from --n-cpu-moe alone, so a placement that pins layers with -ot ...=CPU
    # filed as "all in VRAM" (IQ1_S with 10 layers in RAM, 2026-09-24) and the
    # production Flash-Next entry read 27 in RAM for a shape that pins 37.
    # n_cpu_moe becomes the COUNT of expert layers in RAM from both sources.
    cpu_layers = set(range(cfg.get("n_cpu_moe", 0)))
    for ot in ots:
        for rule in ot.strip("\"'").split(","):
            if "=" not in rule:
                continue
            pat, dev = rule.rsplit("=", 1)
            if dev.strip().upper() != "CPU" or "exps" not in pat:
                continue
            try:
                rx = re.compile(pat)
            except re.error:
                continue
            for i in range(0, 160):
                if rx.search(f"blk.{i}.ffn_up_exps"):
                    cpu_layers.add(i)
    if cpu_layers:
        cfg["n_cpu_moe"] = len(cpu_layers)
    # The split across GPUs. Its ABSENCE on a multi-GPU box is a real cause of
    # OOM: cheyenne piled 18.9 GB onto its 12 GB card for want of one.
    # The ubatch. It is a tuning fact that moves prefill (Flash-Next: 58 -> 332
    # tok/s from -ub 256 -> 2048), so two runs that differ only here must not
    # share a cell - JR keys on it alongside the split (2026-09-24).
    m = re.search(r"(?:^|\s)-ub\s+(\d+)", cmd)
    if m:
        cfg["ubatch"] = int(m.group(1))
    m = re.search(r"--tensor-split\s+(\S+)", cmd)
    if m:
        cfg["tensor_split"] = m.group(1)
    return cfg


def config_json(model, props, cmd, t_ssh=None):
    """The launch config the numbers were measured under - engine-reported
    facts from /props, launch flags parsed from the /running cmd. Absent is
    honest, guessed is not."""
    cfg = {"llama_swap_name": model, "suite": SUITE, "suite_name": SUITE_NAME, "suite_version": SUITE_VERSION}
    gen = props.get("default_generation_settings", {})
    if gen.get("n_ctx"):
        cfg["ctx"] = int(gen["n_ctx"])   # the window ONE REQUEST gets
    if props.get("total_slots"):
        cfg["parallel"] = int(props["total_slots"])
    if props.get("build_info"):
        cfg["engine"] = props["build_info"]
        cfg["engine_name"], cfg["engine_version"] = "llama.cpp", props["build_info"].split("-")[0]
    mods = props.get("modalities")
    if mods is not None:
        cfg["vision"] = bool(mods.get("vision"))
    m = re.search(r"--spec-type\s+(\S+)", cmd)
    if m:
        cfg["spec_type"] = m.group(1)
    m = re.search(r"-ctk\s+(\S+)", cmd)
    if m:
        cfg["cache_k"] = m.group(1)
    m = re.search(r"-ctv\s+(\S+)", cmd)
    if m:
        cfg["cache_v"] = m.group(1)
    cfg.update(placement_from_cmd(cmd))
    # The quant is a property of the FILE THAT WAS LOADED, so it belongs to the
    # measurement - the catalog holds models, not configurations, and asked
    # which quant a run used can only answer "the ones we own".
    #
    # ⚠️ /props.model_path FIRST, the launch cmd only as a fallback. The cmd is
    # whatever /running reports and can come back empty; when it did, no row in
    # a 92-row ledger carried a quant and the tab fell back to the catalog,
    # printing every quant held against every row (2026-08-27). /props is the
    # server describing what it actually opened.
    path = str(props.get("model_path") or "")
    if not path:
        m = re.search(r"--model\s+(\S+\.gguf)", cmd)
        path = m.group(1) if m else ""
    if path:
        # NVFP4 ships in quality tiers (HIGH, HIGHEST) that are separate files
        # and separate catalog quants - capturing a bare "NVFP4" would merge two
        # different setups into one row.
        # ⚠️ The MTP token in the filename is NOT part of the quantisation. It
        # says the file carries a speculative-decoding head, and whether that
        # head was USED is the Mtp column. Writing it into quant duplicated the
        # fact and made an NVFP4 row look like a quant no catalog holds.
        q = re.search(r"(NVFP4(?:-MTP)?(?:-[A-Z]+)?|(?:UD-)?I?Q\d[\w_]*|BF16|F16)",
                      path.split("/")[-1], re.I)
        if q:
            cfg["quant"] = re.sub(r"-MTP", "", q.group(1), flags=re.I)
        t_box = next((x for x in TARGETS.values() if x.get("ssh") == t_ssh), None)
        cfg["source_repo"] = source_repo(t_box, path) if t_box else None
        # The weights this run actually loaded. The catalog cannot supply this -
        # it holds MODELS, so its size is the total of every quant we hold.
        gb = weights_gb_via_ssh(t_ssh, path)
        if gb:
            cfg["weights_gb"] = gb
    return cfg


def assert_gpu(machine, ssh_host, proc="llama-server", container=False):
    """⛔ NEVER PERMIT CPU-ONLY (Jon, 2026-08-27). llama.cpp's dynamic backend
    loader falls back to CPU SILENTLY when the CUDA backend cannot load - on
    pris the b10639 zip shipped without the NVIDIA redist DLLs and the whole
    box benched on CPU while build_info looked perfect. A CPU number in the
    ledger is poison, so proof is mandatory: the serving llama-server process
    must appear in nvidia-smi's compute list on the target. No proof (including
    a failed ssh probe) means NO benchmark - there is no override flag."""
    try:
        if container:
            # A container engine (vLLM, TensorFold Python or Zig): the proof is a process INSIDE a running
            # container holding the weights (>= 10 GiB) in nvidia-smi's compute list. No process names:
            # Mia's Zig engine runs as 'tensorfold-native' and a name list refused it (2026-10-10).
            r = subprocess.run(
                ["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", ssh_host,
                 "pids=$(for c in $(docker ps -q); do docker top $c -eo pid | tail -n +2; done | paste -sd'|'); "
                 "[ -n \"$pids\" ] && nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits "
                 "| awk -F', *' -v re=\"^($pids)$\" '$1 ~ re && $2+0 >= 10240 {print \"GPU-OK\", $1, $2}'"],
                capture_output=True, text=True, errors="replace", timeout=30)
            ok = r.returncode == 0 and "GPU-OK" in r.stdout
        else:
            r = subprocess.run(
                ["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", ssh_host,
                 "nvidia-smi --query-compute-apps=name --format=csv,noheader"],
                capture_output=True, text=True, errors="replace", timeout=30)
            ok = r.returncode == 0 and proc in r.stdout.lower()
    except Exception:
        ok = False
    if not ok:
        raise SystemExit(
            f"⛔ GPU guard: {proc} is NOT in {machine}'s nvidia-smi compute "
            "list (or the ssh probe failed). Refusing to bench - CPU numbers must "
            "never reach the ledger. Fix the backend (CUDA DLLs? driver?), verify "
            "with nvidia-smi, and rerun.")


class WattsSampler:
    """Samples total GPU power draw over ssh once a second while a scenario runs; mean() feeds
    gpu_watts_avg. A pair target's peers (its worker boxes) are sampled in the same tick and summed:
    a pair row once carried the head alone, half the pair (2026-10-10). A tick counts only when
    every box answered. None when the probe fails - omitted, never guessed."""

    def __init__(self, ssh_host):
        peers = next((t.get("peers", []) for t in TARGETS.values() if t.get("ssh") == ssh_host), [])
        self.hosts, self.samples, self._stop = [ssh_host] + list(peers), [], threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    @staticmethod
    def _one(host):
        r = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes", host,
             "nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits"],
            capture_output=True, text=True, errors="replace", timeout=10)
        return sum(float(x) for x in r.stdout.split() if x.strip()) if r.returncode == 0 else None

    def _run(self):
        while not self._stop.is_set():
            try:
                ws = [self._one(h) for h in self.hosts]
                if all(w for w in ws):
                    self.samples.append(sum(ws))
            except Exception:
                pass
            self._stop.wait(1.0)

    def mean(self):
        self._stop.set()
        self._t.join(timeout=5)
        return round(sum(self.samples) / len(self.samples), 1) if self.samples else None


class DiskSampler:
    """Samples sustained disk READ rate on the target while a scenario runs.

    This is what separates "experts live in system RAM" from "experts live on
    the SSD and are paged in every token" - two regimes an order of magnitude
    apart that otherwise look identical in the table. Flash-Next on cheyenne
    read 573 MB/s at 17% iowait for 4 tok/s, while coder-next in RAM on pris
    read nothing for 65.

    ⛔ MEASURED, not computed. It would be easy to infer paging from
    weights > VRAM + RAM, but that is arithmetic dressed as evidence - the same
    mistake as the "won't fit" the matrix used to guess (2026-08-28). The disk
    either thrashed or it did not.

    None when the probe fails: omitted, never guessed.
    """

    # Linux: /proc/diskstats sectors-read (512B units), summed over real disks.
    # Windows: typeperf's one-shot PhysicalDisk read counter.
    _LINUX = ("awk '$3 ~ /^(sd|nvme|vd)/ {s+=$6} END {print s*512}' /proc/diskstats")
    _WIN = ('powershell -Command "(Get-Counter \'\\PhysicalDisk(_Total)\\Disk Read Bytes/sec\')'
            '.CounterSamples[0].CookedValue"')
    # A container engine (vLLM, TensorFold on the Sparks): only the containers' own reads, from their
    # cgroup io.stat (world-readable). The whole disk counted a weight copy as paging: C reading FP8 off
    # its NVMe for D at 460 MB/s labelled a Flash-Next run "paged from SSD" (2026-10-10).
    _CONTAINERS = ("for i in $(docker ps -q --no-trunc); do cat /sys/fs/cgroup/system.slice/docker-$i.scope/io.stat; done"
                   " | grep -o 'rbytes=[0-9]*' | cut -d= -f2 | awk '{s+=$1} END {print s+0}'")

    def __init__(self, ssh_host, os_kind, containers=False):
        self.ssh_host, self.os_kind, self.containers = ssh_host, os_kind, containers
        self.samples, self._prev, self._stop = [], None, threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _probe(self):
        cmd = self._WIN if self.os_kind == "win" else self._CONTAINERS if self.containers else self._LINUX
        r = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes", self.ssh_host, cmd],
            capture_output=True, text=True, errors="replace", timeout=15)
        if r.returncode != 0:
            return None
        for tok in r.stdout.split():
            try:
                return float(tok)
            except ValueError:
                continue
        return None

    def _run(self):
        while not self._stop.is_set():
            try:
                v = self._probe()
                if v is not None:
                    if self.os_kind == "win":
                        # Already a rate.
                        self.samples.append(v / 1e6)
                    else:
                        # A counter: rate is the delta between samples.
                        if self._prev is not None and v >= self._prev:
                            self.samples.append((v - self._prev) / 1e6 / 2.0)
                        self._prev = v
            except Exception:
                pass
            self._stop.wait(2.0)

    def mean_mbs(self):
        self._stop.set()
        self._t.join(timeout=6)
        return round(sum(self.samples) / len(self.samples), 1) if self.samples else None


def weights_gb_via_ssh(ssh_host, path):
    """Size of the loaded weights file, measured on the box. Split GGUFs report
    part 1 only, so sum the set when the name says -00001-of-000NN."""
    if not ssh_host:
        return None
    m = re.match(r"(.*)-\d{5}-of-(\d{5})\.gguf$", path)
    globbed = f"{m.group(1)}-*-of-{m.group(2)}.gguf" if m else path
    try:
        r = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", ssh_host,
             f'python3 -c "import glob,os;print(sum(os.path.getsize(f) for f in glob.glob(r\'{globbed}\')))"'
             if "/" in globbed and not globbed[1:2] == ":" else
             f'powershell -Command "(Get-ChildItem \'{globbed}\' | Measure-Object -Property Length -Sum).Sum"'],
            capture_output=True, text=True, errors="replace", timeout=30)
        b = "".join(c for c in r.stdout if c.isdigit())
        return round(int(b) / 1e9, 2) if b else None
    except Exception:
        return None


# Windows: llama-server's OWN GPU memory, dedicated and SHARED, from the WDDM
# performance counters. This is the per-process figure nvidia-smi cannot give on
# Windows (see vram_gb_via_ssh) and, more importantly, the paging tell: a full
# card on Windows does not error, the driver spills the process into SHARED
# (system) memory and the disk sampler sees nothing because the pages go to RAM.
# Measured 2026-09-24 on the 5090: 0.65 GB shared ran at full speed, 1.1 GB cost
# 35% of generation and 50% of prefill, 4.9 GB cost 93% and 98%. -EncodedCommand
# so the script survives whichever shell Windows OpenSSH hands it to.
_WIN_GPU_PS = (
    '$p=(Get-Process llama-server -ErrorAction SilentlyContinue | Select-Object -First 1).Id; '
    'if($p){ $c=Get-Counter ("\\GPU Process Memory(pid_"+$p+"*)\\Dedicated Usage"),'
    '("\\GPU Process Memory(pid_"+$p+"*)\\Shared Usage"); '
    '$d=($c.CounterSamples | Where-Object {$_.Path -like "*dedicated*"} | Measure-Object -Property CookedValue -Sum).Sum; '
    '$s=($c.CounterSamples | Where-Object {$_.Path -like "*shared*"} | Measure-Object -Property CookedValue -Sum).Sum; '
    '"$d $s" }')


def win_gpu_process_gb(ssh_host):
    """(dedicated_gb, shared_gb) of llama-server on a Windows target, or None."""
    import base64
    enc = base64.b64encode(_WIN_GPU_PS.encode("utf-16-le")).decode()
    try:
        r = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", ssh_host,
             f"powershell -NoProfile -EncodedCommand {enc}"],
            capture_output=True, text=True, errors="replace", timeout=40)
        for line in r.stdout.splitlines():
            parts = line.split()
            if len(parts) == 2 and all(p.replace(".", "").isdigit() for p in parts):
                return round(float(parts[0]) / 1e9, 2), round(float(parts[1]) / 1e9, 2)
    except Exception:
        pass
    return None


def vram_gb_via_ssh(ssh_host):
    """Sum of per-process GPU memory on the target - the models' footprint,
    not the card's total. None (never a guess) when ssh or nvidia-smi fail.

    ⚠️ ALWAYS None ON WINDOWS. nvidia-smi under the WDDM driver model does not
    expose per-process GPU memory and returns [N/A] for every row, so pris
    reports no VRAM at all while cheyenne reports it fine. Checked directly on
    2026-08-28 - this is a driver limitation, not a broken ssh call, so do not
    go looking for it again. Since 2026-09-24 Windows targets get the figure
    from win_gpu_process_gb (the WDDM per-process counters) instead.

    ⛔ Do NOT substitute --query-gpu=memory.used. It works on Windows, but it is
    the WHOLE CARD including the desktop and anything else running - a different
    measurement wearing the same column heading, on a box Jon actively works on.
    Filling the gap that way would make pris and cheyenne silently
    incomparable, which is the exact failure this suite keeps having."""
    try:
        r = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", ssh_host,
             "nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits"],
            capture_output=True, text=True, errors="replace", timeout=30)
        if r.returncode != 0:
            return None
        # [N/A] rows are Windows telling us it cannot answer - not a zero.
        mib = sum(float(x.strip()) for x in r.stdout.splitlines()
                  if x.strip() and "N/A" not in x)
        return round(mib / 1024, 2) if mib else None
    except Exception:
        return None


def llama_bench_args(cmd, os_kind="linux"):
    """llama-bench's argv from a llama-server launch cmd: the same file, placement,
    KV, split and batch, so its pp512/tg128 is the community's number for OUR
    setup. Speculation has no llama-bench flag - its tg128 is always without the
    draft head. Returns (argv, model_path, ctx) or None when the cmd has no model."""
    toks = re.sub(r",\s*env:.*$", "", cmd.strip()).split()
    srv = next((tk for tk in toks if "llama-server" in tk.lower()), toks[0])
    binary = srv[:srv.lower().rindex("llama-server")] + ("llama-bench.exe" if os_kind == "win" else "llama-bench")
    model, ctx, out, i = None, 0, [], 1
    while i < len(toks):
        tk, nx = toks[i], toks[i + 1] if i + 1 < len(toks) else None
        if tk == "--model": model = nx; i += 2
        elif tk == "-c": ctx = int(nx); i += 2
        # llama-bench spells lists its own way: -ot entries by ';', splits and
        # devices by '/', and takes --load-mode over the deprecated -mmp.
        elif tk in ("-ot", "--override-tensor"): out += ["-ot", nx.strip('"').replace(",", ";")]; i += 2
        elif tk in ("--tensor-split", "-ts"): out += ["-ts", nx.replace(",", "/")]; i += 2
        elif tk in ("-ctk", "-ctv", "-ub"): out += [tk, nx]; i += 2
        elif tk in ("--n-cpu-moe", "-ncmoe"): out += ["-ncmoe", nx]; i += 2
        elif tk in ("--device", "-dev"): out += ["-dev", nx.replace(",", "/")]; i += 2
        elif tk == "--load-mode": out += ["--load-mode", nx]; i += 2
        else: i += 1
    if not model:
        return None
    return [binary, "-m", model, "-p", "512", "-n", str(TG_TOKENS), "-ngl", "99", "-fa", "on", "-r", "1", "-o", "json"] + out, model, ctx


def llama_bench(t, model, cmd, ctx, row):
    """One llama-bench pass at the ladder's headline depths, posted as llb_pp512 /
    llb_tg128 under the depth's scenario. The server is unloaded first (the
    bench needs the cards); the next real request reloads it."""
    built = llama_bench_args(cmd, t["os"])
    if not built:
        print("  llama-bench: no --model in the launch cmd, skipped"); return
    argv, path, c = built
    depths = [k for k in LLB_DEPTHS if not (ctx or c) or k * 1024 + 640 <= (ctx or c)]
    if not depths:
        print("  llama-bench: window too small for any depth, skipped"); return
    argv += ["-d", ",".join(str(k * 1024) for k in depths)]
    try:
        http_json(f"{t['api']}/unload", timeout=120)
    except Exception:
        pass
    q = lambda a: "'" + a.replace("'", "'\\''") + "'"
    code, out = _ssh(t, " ".join(q(a) for a in argv), timeout=5400)
    m = re.search(r"\[.*\]", out, re.S)
    if code != 0 or not m:
        print(f"  llama-bench: failed ({out.strip().splitlines()[-1][:160] if out.strip() else 'no output'})"); return
    for r in json.loads(m.group(0)):
        k = int(r.get("n_depth", 0)) // 1024
        scen = f"{k}k"
        if r.get("n_prompt"):
            row("llb_pp512", r["avg_ts"], scen, f"llama-bench pp512 at {k}K depth, no draft head")
        elif r.get("n_gen"):
            row("llb_tg128", r["avg_ts"], scen, f"llama-bench tg128 at {k}K depth, no draft head")
    print("  llama-bench: " + " · ".join(f"{int(r.get('n_depth', 0)) // 1024}k {'pp' if r.get('n_prompt') else 'tg'} {r['avg_ts']:.0f}" for r in json.loads(m.group(0))))


def wait_quiet(t, poll=30):
    """Hold the run until no box of the target (the head and a pair's peers) is copying weights or
    loading an image: rsync, docker load / save. A copy over the pair link or from the archive
    halved a 2026-10-10 run's decode (20 tok/s against 56 clean). Linux targets only."""
    if t.get("os") != "linux" or not t.get("ssh"):
        return
    hosts = [t["ssh"]] + list(t.get("peers", []))
    probe = "ps -eo args | grep -E '^(rsync|docker load|docker save)' | wc -l"
    said = False
    while True:
        busy = []
        for h in hosts:
            r = subprocess.run(["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes", h, probe],
                               capture_output=True, text=True, timeout=20)
            if r.returncode == 0 and r.stdout.strip() not in ("", "0"):
                busy.append(h.split("@")[-1])
        if not busy:
            return
        if not said:
            print(f"  waiting: copies running on {', '.join(busy)} - the run starts when every box is quiet")
            said = True
        time.sleep(poll)


def bench_model(api, model, machine, parallel_n, ssh_host, submit, os_kind="linux", depths=DEPTHS, llb=False):
    rows = []
    now = lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

    def row(metric, value, scenario, note=None):
        r = {"model": model, "metric": metric, "value": round(value, 3),
             "machine": machine, "scenario": scenario,
             "configJson": json.dumps(cfg), "source": SOURCE,
             "notes": note, "capturedAt": now()}
        rows.append(r)
        submit(r)  # live: each row posts the moment it exists

    t_box = next((t for t in TARGETS.values() if t["api"] == api), {})
    vllm = t_box.get("engine") == "vllm"
    wait_quiet(t_box)
    print("  warm-up (may include model load)...")
    run_one(api, model, warmup_prompt(), 16)
    # The hello above loads the model; it does not warm the long-prompt path. The first ~4K
    # prefill carries one-off kernel compilation / graph capture, and the 4k rung was paying
    # for it (GLM-5.3-Flash on A+B read 1,022 tok/s at 4k, ~1,600 at every deeper rung).
    print("  warm-up 2: one ~4K prompt, cache off, nothing recorded...")
    if vllm:
        run_vllm(api, model, lambda: [ladder_prompt(4)], 16, timeout=3600)
    else:
        run_one(api, model, ladder_prompt(4), 16, cache_prompt=False, timeout=3600)
    engine = engine_of(api) if vllm else "llama-server"
    assert_gpu(machine, ssh_host, engine if vllm else "llama-server", container=vllm)  # every model, after load, before any row
    if vllm:
        launch_cmd = ""
        cfg = config_vllm(t_box, model)
    else:
        try:
            props = http_json(f"{api}/upstream/{model}/props", timeout=60)
        except Exception:
            props = {}
        launch_cmd = running_cmd(api, model)
        cfg = config_json(model, props, launch_cmd, ssh_host)

    # The ladder: every rung the window allows. A rung that does not fit is
    # skipped, not refused - the window is a config fact, not a fit verdict.
    ctx = cfg.get("ctx") or 0
    rungs = [k for k in depths if not ctx or k * 1000 + TG_TOKENS + 512 <= ctx]
    top = rungs[-1] if rungs else None
    # A rung that crashes the server ends the ladder: every later prompt would go
    # to a dead or reloading server and fail with no cause of its own (the matrix
    # shows those rungs as "not reached"). The 4-stream step is skipped too.
    crashed = False
    for k in rungs:
        scen = f"{k}k"
        # The top rung of a window aims UNDER it: the chars-per-token cut runs
        # ~2.5% hot at depth and a 32k rung on a 32K slot landed at 32,811 tokens
        # and was refused (2026-09-25). The row's note carries the served count.
        target = min(k * 1000, ctx - TG_TOKENS - 1024) if ctx else k * 1000
        if vllm:
            target = vllm_fit(api, model, ladder_prompt, target / 1000, ctx) * 1000
        prompt = ladder_prompt(target / 1000)
        # Watts on every rung; disk and the Windows paging tell on the deepest
        # rung only, where paging shows.
        watts = WattsSampler(ssh_host) if ssh_host else None
        disk = DiskSampler(ssh_host, os_kind, containers=vllm) if k == top and ssh_host else None
        try:
            if vllm:
                res = run_vllm(api, model, lambda: [ladder_prompt(target / 1000)], TG_TOKENS, timeout=3600)[0]
            else:
                res = run_one(api, model, prompt, TG_TOKENS, cache_prompt=False, timeout=3600)
        except Exception:
            if watts: watts.mean()
            if k == rungs[0]:
                raise   # the first rung failing is a load problem - the caller judges it
            # A rung that DIED is a result, not a gap (Jon, 2026-09-25: "add an
            # indication if it has been tested and failed"). Evidence-gated like a
            # refusal: rung_failed posts only with the engine's (or the kernel's)
            # own cause line from the box's logs; nothing else is written.
            why = rung_cause(ssh_host)
            if why:
                row("rung_failed", 1, scen, why)
                print(f"  {scen}: FAILED - {why}")
            else:
                print(f"  {scen}: failed - no cause in the box's logs, nothing recorded")
            crashed = True
            later = [f"{x}k" for x in rungs if x > k]
            if later:
                print(f"  {', '.join(later)} + x4: not reached - the ladder stops at a crash")
            break
        w = watts.mean() if watts else None
        d = disk.mean_mbs() if disk else None
        if res["output_tokens"] < 2:
            print(f"  {scen}: dropped (output_tokens={res['output_tokens']})")
            continue
        note = f"{res['prompt_tokens']:,}-token prompt, tg{TG_TOKENS}"
        if res.get("cache_n"):
            note += f", cache_n={res['cache_n']}"   # should be 0: a cached rung is not cold
        if res["prompt_tps"]:
            row("prompt_tok_s", res["prompt_tps"], scen, note)
        if res["gen_tps"]:
            row("gen_tok_s", res["gen_tps"], scen, note)
        if res["ttft_s"]:
            row("ttft_s", res["ttft_s"], scen, note)
        if res["accept"] is not None:
            row("draft_accept", res["accept"], scen, "accepted / drafted, from the server's timings")
        if w:
            row("gpu_watts_avg", w, scen, "all GPUs summed across every box of the target (a pair: both), 1s samples via nvidia-smi over ssh")
        if d is not None:
            # Sustained disk READ during the request: "experts in RAM" and
            # "experts paged from SSD every token" look identical otherwise.
            if cfg.get("engram_offload") == "ssd":
                # DeepSeek-V4.1's recipe serves its 190 GB engram table from a FILE on the NVMe by design;
                # the weights are all in VRAM. Its reads are a fact of the recipe, not weights paging, so
                # they post apart and the matrix places the run by its weights (2026-10-10: "paged from SSD").
                row("engram_read_mbs", d, scen, "sustained disk read during the request (the file-backed engram table), sampled over ssh")
            else:
                row("disk_read_mbs", d, scen, "sustained disk read during the request, sampled over ssh")
        shared = None
        if k == top and os_kind == "win" and ssh_host:
            g = win_gpu_process_gb(ssh_host)
            if g:
                shared = g[1]
                row("vram_shared_gb", shared, scen,
                    "llama-server's GPU memory in SHARED (system) memory with the model loaded - "
                    "WDDM GPU Process Memory counters over ssh; the paging tell on Windows")
        print(f"  {scen}: {res['prompt_tokens']:,} tok, {res['prompt_tps'] or 'n/a'} prefill, "
              f"{res['gen_tps'] or 'n/a'} gen, ttft {res['ttft_s'] or 'n/a'} s"
              + (f", accept {res['accept']:.2f}" if res['accept'] is not None else "")
              + (f", {w} W" if w else "") + (f", {d} MB/s disk" if d else "")
              + (f", {shared} GB shared" if shared is not None else ""))

    # Concurrency (v5): parallel_n streams at tg1024, at the deepest rung where
    # every stream fits its prompt + output in the window one request gets - 32K
    # down to 4K. v4 ran this stage tg128 at a fixed 4K: ~2 s of decode per stream
    # against a queue of serialized prefills, so on an engine that staggers streams
    # each one decoded essentially alone and "each" was solo speed inside a
    # 4-stream test (2026-10-24: "67 each / 30 total" on the coding pair - 67 was
    # its solo figure, real telemetry measured at zero overlap). A decode window
    # that outlasts the prefill queue is what makes `each` mean "per stream with
    # N streams flying". 4K stays the floor; four 32K prompts still overflow a
    # 4-slot 32K window's KV and read "no clean streams" (2026-09-25).
    x_depth = next((k for k in (32, 16, 8, 4)
                    if not ctx or k * 1000 + TG_X + 1024 <= ctx), 4) if rungs and not crashed else None
    if x_depth:
        scen = f"{x_depth}k_x{parallel_n}"
        results, threads = [None] * parallel_n, []
        watts = WattsSampler(ssh_host) if ssh_host else None
        start = time.time()
        for i in range(parallel_n):
            def work(idx=i):
                try:
                    results[idx] = run_one(api, model, ladder_prompt(x_depth, stream=idx),
                                           TG_X, cache_prompt=False, timeout=3600)
                except Exception as e:
                    results[idx] = {"error": str(e)}
            threads.append(threading.Thread(target=work))
        if vllm:
            try:
                results = run_vllm(api, model, lambda: [ladder_prompt(x_depth, stream=i) for i in range(parallel_n)],
                                   TG_X, timeout=3600)
                start = time.time() - results[0]["wall"]
            except Exception as e:
                results = [{"error": str(e)}] * parallel_n
            threads = []
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        wall = time.time() - start
        par_w = watts.mean() if watts else None
        good = [r for r in results if r and not r.get("error")
                and r["output_tokens"] >= 2 and r["gen_tps"]]
        if good:
            each = sum(r["gen_tps"] for r in good) / len(good)
            gtoks = sum(r["output_tokens"] for r in good)
            agg = gtoks / wall
            # The overlap check: summed per-stream decode time over the run's wall.
            # ~N means the streams really decoded together; ~1 means the engine
            # serialized them and `each` is (near-)solo speed wearing a concurrency
            # label. The row says so out loud - a reader must never have to redo
            # this arithmetic to trust a cell (2026-10-24).
            overlap = (gtoks - len(good)) / (each * wall) if each and wall else None
            note = (f"mean of {len(good)}/{parallel_n} streams, {x_depth}k prompt each, tg{TG_X}"
                    + (f", overlap x{overlap:.1f}" if overlap else ""))
            if len(good) >= 2 and (each > agg or (overlap and overlap < 1.5)):
                note += (" - WARNING the streams' decode windows barely overlapped: the engine"
                         " serialized the run, so each reads near-solo and the aggregate is the"
                         " honest figure for concurrent use")
            row("gen_tok_s", each, scen, note)
            row("gen_tok_s_aggregate", agg, scen, f"{len(good)}/{parallel_n} streams, {wall:.1f}s wall incl. prefill")
            if par_w:
                row("gpu_watts_avg", par_w, scen, "all GPUs summed across every box of the target (a pair: both), 1s samples via nvidia-smi over ssh")
            print(f"  {scen}: {len(good)}/{parallel_n} clean streams, each {each:.1f} / total {agg:.1f} tok/s"
                  + (f", overlap x{overlap:.1f}" if overlap else "")
                  + (f", {par_w} W" if par_w else ""))
        else:
            print(f"  {scen}: no clean streams")

    vram = vram_gb_via_ssh(ssh_host)
    if vram:
        row("vram_gb", vram, None, "per-process sum via nvidia-smi over ssh")
    if llb and ssh_host:
        t_llb = next((t for t in TARGETS.values() if t["ssh"] == ssh_host), None)
        if t_llb:
            llama_bench(t_llb, model, launch_cmd, cfg.get("ctx") or 0, row)
        print(f"  vram: {vram} GB")
    elif os_kind == "win" and ssh_host and (g := win_gpu_process_gb(ssh_host)):
        # WDDM per-process dedicated memory - llama-server's own footprint on
        # the card, the same claim the nvidia-smi figure makes on Linux.
        row("vram_gb", g[0], None, "llama-server's dedicated GPU memory via the WDDM GPU Process Memory counter over ssh")
        print(f"  vram: {g[0]} GB dedicated, {g[1]} GB shared")
    else:
        print("  vram: unavailable (ssh/nvidia-smi) - omitted, not guessed")
    return rows


# ---------------------------------------------------------------- gap-fill
# The matrix's question is every setup x every machine; a cell is a number,
# a wont_fit verdict, or honestly untested. --gap-fill runs the whole loop
# for one target so an agent only executes, never derives:
#   find the setup in another box's ARCHIVE MASTER -> translate paths/splits
#   -> temp entry on the DEPLOYED yaml only -> bench (or record wont_fit
#   with the real load error) -> restore the master -> reload the brain.

# ⚠️ errors="replace" on every subprocess that reads a box's output. Pris is
# Windows and its console is not UTF-8: a single 0xe6 byte in nvidia-smi/log
# output raised UnicodeDecodeError and killed a whole run mid-measurement
# (2026-08-28). A mangled character in a log line is survivable; losing the run
# is not.
def _ssh(t, cmd, timeout=60):
    r = subprocess.run(["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes",
                        t["ssh"], cmd], capture_output=True, text=True, errors="replace", timeout=timeout)
    return r.returncode, r.stdout


def _scp(src, dst, timeout=120):
    r = subprocess.run(["scp", "-q", src, dst], capture_output=True, text=True, errors="replace", timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"scp {src} -> {dst}: {r.stderr.strip()}")


def wait_settled(api, timeout=900):
    """Block until llama-swap has no model mid-load. A -watch-config reload
    fires the on_startup preload, and a swap requested DURING that load gets
    a 500 - observed by opencode's manual gap-fill 2026-08-27. A transient
    500 must never be read as a fit verdict."""
    end = time.time() + timeout
    stable = 0
    empty = 0
    while time.time() < end:
        try:
            recs = http_json(f"{api}/running", timeout=15).get("running", [])
            # ⚠ An EMPTY list is NOT INSTANTLY settled: right after a
            # watch-config reload llama-swap has dropped everything and the
            # preload has not appeared yet, and all([]) is True - which let the
            # old check return immediately and every request 500 (2026-08-27,
            # opencode's run recorded two false wont_fit verdicts that way).
            # Require a ready model seen twice in a row.
            if recs and all(r.get("state") == "ready" for r in recs):
                stable += 1
                if stable >= 2:
                    return
                empty = 0
            elif not recs:
                # ...but a PERSISTENTLY empty box is settled: it is idle and
                # nothing is coming. Without this the wait ran the full 900s
                # whenever the model being tested was the only one asked for
                # and it failed to load - which is exactly the case a refusal
                # test creates. It read as a hang (2026-08-28).
                empty += 1
                stable = 0
                if empty >= 6:      # ~30s of a genuinely idle box
                    return
            else:
                stable = 0
                empty = 0
        except Exception:
            stable = 0
            empty = 0
        time.sleep(5)
    print("  ⚠ llama-swap still loading after settle timeout - proceeding anyway")


def find_source_entry(sid, exclude):
    """Locate the setup id in another box's archive master; return the cmd
    resolved against that master's macros (${llama-server} stays - the target
    supplies its own)."""
    for name, t in TARGETS.items():
        if name == exclude or not os.path.exists(t["master"]):
            continue
        text = open(t["master"]).read()
        m = re.search(rf'^  "{re.escape(sid)}":\n(.*?)(?=^  "|^[a-z#=]|\Z)', text,
                      re.M | re.S)
        if not m:
            continue
        c = re.search(r"cmd: \|\n((?:      .*\n)+)", m.group(1))
        if not c:
            continue
        cmd = " ".join(line.strip() for line in c.group(1).splitlines())
        for mac, val in re.findall(r'^  "([\w-]+)":\s*"([^"]+)"$', text, re.M):
            if mac != "llama-server":
                cmd = cmd.replace("${%s}" % mac, val)
        return name, cmd
    raise SystemExit(f"⛔ '{sid}' found in no archive master - a gap-fill reuses an "
                     "existing setup id verbatim; check the id against the source box's yaml.")


def build_temp_entry(t, sid, cmd):
    """Rewrite a source cmd for the target box: its model root, its tuned
    split macro, everything else verbatim. Returns (entry_text, cfg_facts)."""
    toks = cmd.split()
    if "--n-cpu-moe" in toks and t["no_offload"]:
        raise SystemExit(f"⛔ '{sid}' spills experts and {t['machine']} has the zero-offload "
                         "rule. That is a policy wont_fit - enter it over MCP, do not attempt it.")
    out, facts, i = [], {"vision": False}, 0
    model_dir = None
    while i < len(toks):
        tk = toks[i]
        if tk in ("--model", "--mmproj"):
            src_path = toks[i + 1]
            model_dir, fname = src_path.rstrip("/").split("/")[-2:]
            tgt = f"{t['model_root']}/{model_dir}/{fname}"
            code, _ = _ssh(t, f"test -f '{tgt}'" if t["os"] == "linux"
                           else f'if exist "{tgt}" (exit 0) else (exit 1)')
            if code != 0:
                raise SystemExit(f"⛔ {t['machine']} is missing {tgt} - copy the weights "
                                 "first (and backfill the archive if it lacks them too).")
            if tk == "--mmproj":
                facts["vision"] = True
            if tk == "--model":
                # ⚠️ The QUANT of the file we are about to attempt. For a REFUSAL
                # this is the only source there is - the server never loaded, so
                # /props cannot be asked. Without it a wont_fit verdict cannot be
                # matched to the row it belongs to: quant is part of a row's
                # identity, and the four verdicts recorded before this went
                # invisible in the matrix (2026-08-28).
                # Same tier-aware pattern as the loaded-run path below: NVFP4
                # ships as HIGH / HIGHEST files, and a bare "NVFP4" here keyed
                # the refusal to a row nothing else uses, so cheyenne's
                # HIGHEST cell read "untested" over a recorded verdict (2026-09-24).
                q = re.search(r"(NVFP4(?:-MTP)?(?:-[A-Z]+)?|(?:UD-)?I?Q\d[\w_]*|BF16|F16)", fname, re.I)
                if q:
                    facts["quant"] = re.sub(r"-MTP", "", q.group(1), flags=re.I)
            out += [tk, tgt]
            i += 2
            continue
        if tk == "--tensor-split":   # source box's split never transfers
            i += 2
            continue
        if tk == "-a":
            out += ["-a", sid]
            i += 2
            continue
        if tk == "-c":
            facts["ctx"] = int(toks[i + 1])
        if tk == "-ctk":
            facts["cache_k"] = toks[i + 1]
        if tk == "-ctv":
            facts["cache_v"] = toks[i + 1]
        if tk == "--spec-type":
            facts["spec_type"] = toks[i + 1]
        out.append(tk)
        i += 1
    split = t["split_macros"].get(model_dir or "")
    if split:
        out.append(split)
    flags = " ".join(tk for tk in out if tk != "${llama-server}")
    entry = (f'\n  # ⚠ TEMP gap-fill entry (bench.py --gap-fill). The archive master does\n'
             f'  # not carry it; the restore step erases it. Never let it outlive the run.\n'
             f'  "{sid}":\n'
             f'    name: "{sid} (gap-fill)"\n'
             f'    description: "TEMP gap-fill - exact setup of the source box"\n'
             f'    cmd: |\n'
             f'      ${{llama-server}}\n'
             f'      {flags}\n'
             f'    ttl: 0\n')
    return entry, facts


def raw_entry(t, sid, cmd):
    """A temp entry from a launch command the box once ran (journal
    'Executing start command'), verbatim except the port: for re-benching
    retired placements on the SAME box under a new suite version."""
    # ⚠️ The journal line is "<cmd>, env: [...]" - llama-swap appends the entry's
    # env block after a comma. Deployed verbatim, llama-server refused every
    # entry with "invalid argument: env:" and eight refusals were recorded
    # against the 3.6 Q4 before the run was stopped (2026-09-25).
    cmd = re.sub(r",\s*env:.*$", "", cmd.strip())
    toks = cmd.split()
    # The binary is everything up to and including the llama-server token: on
    # pris that is a powershell wrapper (llm-run.ps1) in front of the .exe.
    bi = next((i for i, tk in enumerate(toks) if "llama-server" in tk.lower()), 0)
    binary, flags = " ".join(toks[:bi + 1]), toks[bi + 1:]
    for i, tk in enumerate(flags):
        if tk == "--port":
            flags[i + 1] = "${PORT}"
        if tk in ("--model", "--mmproj", "-md"):
            tgt = flags[i + 1]
            code, _ = _ssh(t, f"test -f '{tgt}'" if t["os"] == "linux"
                           else f'if exist "{tgt}" (exit 0) else (exit 1)')
            if code != 0:
                raise SystemExit(f"⛔ {t['machine']} is missing {tgt} - copy the weights first.")
        if tk == "-a":
            flags[i + 1] = sid
    if "--port" not in flags:
        flags = ["--port", "${PORT}", "--host", "127.0.0.1", "--metrics"] + flags
    return (f'\n  # ⚠ TEMP re-bench entry (bench.py --entries). The master does not carry it;\n'
            f'  # the restore step erases it. Never let it outlive the run.\n'
            f'  "{sid}":\n'
            f'    name: "{sid} (re-bench)"\n'
            f'    description: "TEMP re-bench - a retired placement, launch cmd from the journal"\n'
            f'    cmd: |\n'
            f'      {binary}\n'
            f'      {" ".join(flags)}\n'
            f'    ttl: 0\n')


def read_deployed(t):
    local = f"/tmp/gapfill-{t['machine']}.yaml"
    _scp(f"{t['ssh']}:{t['yaml']}", local)
    return local, open(local).read()


def write_deployed(t, text):
    local = f"/tmp/gapfill-{t['machine']}.yaml"
    open(local, "w").write(text)
    _scp(local, f"{t['ssh']}:{t['yaml']}")


def cfg_from_yaml(t, sid):
    """The setup as the DEPLOYED yaml declares it, for a model that never loaded.

    ⛔ A refusal cannot be described from /props - the server exited. The yaml
    entry is the only witness to what was attempted, and without it a verdict
    lands with an empty config: quant null, ctx null, everything null. That is
    not a verdict about a setup, it is a verdict about a name, and it matches no
    row in the matrix (2026-08-28, verdict id=10).
    """
    try:
        _, text = read_deployed(t)
    except Exception:
        return {"llama_swap_name": sid}
    # The entry runs from its quoted id to the next top-level key.
    m = re.search(rf'^\s*"{re.escape(sid)}":\s*$(.*?)(?=^\s*"[^"]+":\s*$|^[a-z_]+:)',
                  text, re.M | re.S)
    if not m:
        return {"llama_swap_name": sid}
    body = m.group(1)
    cfg = {"llama_swap_name": sid, "suite": SUITE, "suite_name": SUITE_NAME, "suite_version": SUITE_VERSION}
    mm = re.search(r"--model\s+(\S+\.gguf)", body)
    if mm:
        q = re.search(r"(NVFP4(?:-MTP)?(?:-[A-Z]+)?|(?:UD-)?I?Q\d[\w_]*|BF16|F16)", mm.group(1).split("/")[-1], re.I)
        if q:
            cfg["quant"] = re.sub(r"-MTP", "", q.group(1), flags=re.I)   # tier-aware, as above
    for flag, key, cast in (("-c", "ctx", int), ("-ctk", "cache_k", str),
                            ("-ctv", "cache_v", str), ("--spec-type", "spec_type", str),
                            ("--parallel", "parallel", int)):
        mm = re.search(rf"{re.escape(flag)}\s+(\S+)", body)
        if mm:
            try:
                cfg[key] = cast(mm.group(1))
            except ValueError:
                pass
    cfg["vision"] = "--mmproj" in body
    cfg.setdefault("parallel", 4)
    # The placement, so a refusal lands on the row it was refused FOR - JR keys
    # rows on layers-in-RAM since 0.9.0.168 (and on split / ubatch as tuning).
    cfg.update(placement_from_cmd(body))
    return cfg


def is_engine_error(ln):
    """A load-refusal cause line is the ENGINE's (cudaMalloc, alloc, OOM) - never
    llama-swap's own proxy chatter. 'http: proxy error: connection refused' is
    what every failed start looks like from the outside and it filed eight
    refusals for entries that never parsed their command line (2026-09-25)."""
    if re.search(r"http: proxy error|non-200 response|invalid argument", ln, re.I):
        return False
    return re.search(r"cuda|alloc|memory|failed|error", ln, re.I) is not None


def clean_log_line(ln):
    """One log line, stripped to the message.

    Verdict reasons are read in a table cell and mirrored to the cloud, so they
    carry no journal furniture: no syslog date, no hostname, no pid, and none of
    the engine's own 0.00.714.486-style uptime/level prefix. Absolute paths go
    too - they are storage topology, and a reason should say what failed, not
    where the file lives (2026-08-28: reasons were arriving as
    "Aug 28 15:55:09 cheyenne llama-swap[2000]: 0.00.849 E srv load_model: ...").
    """
    ln = ln.strip()
    # syslog: "Aug 28 15:55:09 host unit[123]: "
    ln = re.sub(r"^[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}\s+\S+\s+\S+?\[\d+\]:\s*", "", ln)
    # llama.cpp: "0.00.714.486 E ggml_backend..." / "... W srv  "
    ln = re.sub(r"^\d+\.\d+\.\d+\.\d+\s+[EWID]\s+(?:srv|cmn|common\S*|load)?\s*", "", ln)
    # llama-swap: "[WARN] ", "[DEBUG] "
    ln = re.sub(r"^\[[A-Z]+\]\s*", "", ln)
    # absolute paths -> the filename alone
    ln = re.sub(r"[A-Za-z]:[\\/][^\s'\"]+[\\/]|/(?:[\w.\-]+/)+", "", ln)
    return ln.strip()


def rung_cause(ssh_host):
    """Why a ladder rung died, from the box's own logs: the engine's cudaMalloc /
    allocation line, or the kernel OOM killer (systemd logs it under the unit).
    None when the logs carry no such line - then nothing is recorded."""
    t = next((t for t in TARGETS.values() if t["ssh"] == ssh_host), None)
    if t is None:
        return None
    try:
        _, log = _ssh(t, t["log_cmd"], timeout=30)
    except Exception:
        return None
    lines = [clean_log_line(ln) for ln in log.splitlines()]
    causes = [ln for ln in lines if ln and re.search(
        r"cudamalloc|out of memory|failed to allocate|oom.kill|segmentation fault", ln, re.I)]
    picked = []
    for ln in causes[-3:]:
        if ln not in picked:
            picked.append(ln)
    return "; ".join(picked)[:400] or None


def record_refusal(t, sid, err, mcp, ids, cfg_hint=None):
    """Evidence-gated: a verdict only when the box's own logs carry a load
    error (tested-data-only house rule). Returns True if recorded."""
    _, log = _ssh(t, t["log_cmd"], timeout=30)
    lines = [clean_log_line(ln) for ln in log.splitlines()
             if is_engine_error(ln)]
    lines = [ln for ln in lines if ln]
    if not lines:
        print(f"  INCONCLUSIVE ({err}) - no load error in {t['machine']}'s logs; "
              "nothing recorded, the cell stays untested.")
        return False
    # ⚠️ Keep the line that says WHY, not the last line before the exit. Engines
    # log the cause first and the consequence three times after it, so taking
    # the tail recorded "failed to load model" and "upstream command exited
    # prematurely" - true, and useless. That verdict got read as a VRAM ceiling
    # when the actual line said 18.9 GB was being put on a 16 GB card by a
    # config with no tensor split (2026-08-28).
    causes = [ln for ln in lines if re.search(
        r"cudamalloc|out of memory|failed to allocate|unable to allocate|"
        r"cannot allocate|no such file|unknown model architecture|unsupported", ln, re.I)]
    # ⛔ NO CAUSE, NO VERDICT. Falling back to any line mentioning "failed" let
    # llama-swap's own "upstream command exited prematurely" become a verdict -
    # that says the child process died, NOT why, and it is emitted whatever the
    # reason. On 2026-08-28 it produced a wont_fit for a config that may well
    # run, because pris does not capture the upstream llama-server's stderr at
    # all, so no real error was ever available to read.
    # The house rule is the point: a refusal with no load error in the box's own
    # logs is INCONCLUSIVE and nothing is recorded.
    if not causes:
        print(f"  INCONCLUSIVE ({err}) - {t['machine']}'s logs show the process exited "
              "but carry no load error, so nothing is recorded. The cell stays untested. "
              "(If this box does not capture the engine's stderr, fix that first - a "
              "verdict needs the engine's own words.)")
        return False
    picked, seen = [], set()
    for ln in causes[-3:]:
        if ln not in seen:
            seen.add(ln)
            picked.append(ln)
    reason = "; ".join(picked)[:400]
    print(f"  load refused -> recording fit verdict: {reason}")
    if cfg_hint is None:
        cfg_hint = cfg_from_yaml(t, sid)
    if mcp:
        mid = resolve_model_id(ids, sid)
        if mid:
            mcp.call("models_record_wont_fit", {
                "modelId": mid, "machine": t["machine"],
                "configJson": json.dumps(cfg_hint or {"llama_swap_name": sid}),
                "reason": reason, "source": SOURCE})
    return True


def gap_fill(tname, sids, parallel, mcp, ids, submit, depths=DEPTHS, raw_cmds=None, llb=False):
    """raw_cmds: {sid: launch cmd} re-benches retired placements of THIS box
    (--entries); without it the setups come from another box's master."""
    t = TARGETS[tname]
    served_before = set(served_models(t["api"]))
    entries, all_facts = [], {}
    for sid in sids:
        if sid in served_before:
            sys.exit(f"'{sid}' is already served on {tname} - that is a plain "
                     f"bench, not a gap: bench.py --target {tname} --models {sid}")
        if raw_cmds is not None:
            entry = raw_entry(t, sid, raw_cmds[sid])
            all_facts[sid] = {}
            entries.append(entry)
            print(f"[{sid}] launch cmd from the journal; weights verified on {t['machine']}")
            continue
        src, cmd = find_source_entry(sid, tname)
        entry, facts = build_temp_entry(t, sid, cmd)
        all_facts[sid] = facts
        entries.append(entry)
        print(f"[{sid}] source: {src}'s master; weights verified on {t['machine']}")

    # Temp entries go INTO the models: section - before selectors:/hooks: if
    # the yaml has them, else at EOF. Deployed copy only, never the master.
    local, text = read_deployed(t)
    # ⛔ RESTORE WHAT WAS THERE, not the master: the snapshot is the exact
    # pre-run yaml, and a stale master path (one moved on 2026-09-22) would
    # otherwise leave every temp entry
    # deployed. The master is still consulted for gap-fill SOURCES.
    snapshot = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results",
                            f".yaml-before-{tname}-{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    open(snapshot, "w").write(text)
    m = re.search(r"^(?:selectors|hooks):", text, re.M)
    pos = m.start() if m else len(text)
    write_deployed(t, text[:pos] + "".join(entries) + "\n" + text[pos:])
    print(f"temp entries deployed to {t['machine']} (watch-config reloads; the "
          "loaded model is evicted - expected)")
    try:
        deadline = time.time() + 30
        while time.time() < deadline:
            if all(s in served_models(t["api"]) for s in sids):
                break
            time.sleep(3)
        else:
            raise SystemExit("⛔ temp entries never appeared in /v1/models - deployed "
                             "yaml may be invalid. Restoring the master.")
        # The reload just fired the preload hook - let that load finish
        # before asking for a different model, or the swap 500s.
        wait_settled(t["api"])
        all_rows = []
        for sid in sids:
            print(f"\n[{sid}]")
            try:
                all_rows.extend(bench_model(t["api"], sid, t["machine"], parallel, t["ssh"], submit, t["os"], depths, llb))
                continue
            except SystemExit:
                raise                       # the GPU guard stays fatal
            except Exception as first_err:
                # Transient (mid-load 500, health timeout) vs real refusal:
                # settle and retry ONCE before believing it.
                print(f"  attempt failed ({first_err}) - settling, then one retry")
            wait_settled(t["api"])
            try:
                all_rows.extend(bench_model(t["api"], sid, t["machine"], parallel, t["ssh"], submit, t["os"], depths, llb))
                continue
            except SystemExit:
                raise
            except Exception as e:
                _, log = _ssh(t, t["log_cmd"], timeout=30)
                err = [ln for ln in log.splitlines()
                       if is_engine_error(ln)][-3:]
                # ⛔ A VERDICT NEEDS EVIDENCE. Without a load error in the
                # box's own logs, a failure is INCONCLUSIVE - an HTTP 500
                # says the swap was refused, not that the setup cannot fit,
                # and recording it as wont_fit launders a timing hiccup into
                # a permanent "cannot serve" (2026-08-27: two such rows had
                # to be deleted). Leave the cell untested and say so.
                if not err:
                    print(f"  INCONCLUSIVE ({e}) - no load error in {t['machine']}'s logs, "
                          "so nothing is recorded. The cell stays untested; rerun when the box is idle.")
                    continue
                reason = "; ".join(err)[:400]
                print(f"  load refused -> recording fit verdict: {reason}")
                if mcp:
                    f = all_facts[sid] or cfg_from_yaml(t, sid)
                    cfg = {"llama_swap_name": sid, "parallel": 4, "suite": SUITE, "suite_name": SUITE_NAME,
                           "suite_version": SUITE_VERSION, **{k: v for k, v in f.items()}}
                    mid = resolve_model_id(ids, sid)
                    if mid:
                        # Verdicts have their own table - never the ledger.
                        mcp.call("models_record_wont_fit", {
                            "modelId": mid, "machine": t["machine"],
                            "configJson": json.dumps(cfg),
                            "reason": reason, "source": SOURCE})
    finally:
        here = os.path.dirname(os.path.abspath(__file__))
        out = os.path.join(here, "results", f"{tname}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(out, "w") as f:
            json.dump({"target": tname, "machine": t["machine"], "rows": all_rows, "unsubmitted": []}, f, indent=2)
        print(f"\n{t['machine']}: {len(all_rows)} row(s) -> {out}")
        write_deployed(t, text)
        print(f"\n{t['machine']}'s yaml restored from the pre-run snapshot ({snapshot})")
        if t["preferred"]:
            print(f"reloading {t['preferred']} (the box's resident model)...")
            try:
                http_json(f"{t['api']}/upstream/{t['preferred']}/props", timeout=REQUEST_TIMEOUT)
            except Exception:
                print("  (reload request failed - the next real request will load it)")


# ------------------------------------------------------------ orchestration

def served_models(api):
    return [m["id"] for m in http_json(f"{api}/v1/models", timeout=30)["data"]
            if m["id"] not in SKIP_IDS]


def probe_fleet():
    """target -> served model list, for every box that answers."""
    fleet = {}
    for tname, t in TARGETS.items():
        try:
            fleet[tname] = served_models(t["api"])
        except Exception:
            print(f"({tname} unreachable - skipped)")
    return fleet


def interactive_pick(fleet):
    # ⚠️ "one box at a time" was removed 2026-08-28: different boxes run
    # concurrently and a sweep takes half the wall clock. Only a SECOND run
    # against the SAME box is refused, by the per-target lockfile.
    print("\nAvailable (4-slot catalog; different boxes may run concurrently):")
    index = []
    for tname, models in fleet.items():
        for m in models:
            index.append((tname, m))
            print(f"  {len(index):2}. {tname:9} {m}")
    sel = input("\nSelect numbers (e.g. 1,3-5), or 'all': ").strip().lower()
    if sel in ("all", "a", "*", ""):
        return fleet
    picked = set()
    for part in sel.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            picked.update(range(int(lo), int(hi) + 1))
        elif part:
            picked.add(int(part))
    chosen = {}
    for n in sorted(picked):
        tname, m = index[n - 1]
        chosen.setdefault(tname, []).append(m)
    return chosen


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", choices=sorted(TARGETS), help="one box; omit with --all for the fleet")
    ap.add_argument("--models", help="comma-separated subset (requires --target)")
    ap.add_argument("--all", action="store_true", help="every model on every reachable box")
    ap.add_argument("--gap-fill", action="store_true",
                    help="fill matrix gaps on --target: the setups in --models are pulled "
                         "from another box's archive master, temp-deployed, benched (or "
                         "recorded wont_fit with the real load error), then the master is restored")
    ap.add_argument("--entries", metavar="JSON",
                    help="re-bench retired placements of --target: a JSON {id: launch cmd} "
                         "(journal 'Executing start command'), temp-deployed like --gap-fill")
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--depths", default=None,
                    help="comma-separated rungs in K tokens (default: the full ladder "
                         + ",".join(map(str, DEPTHS)) + ")")
    ap.add_argument("--llama-bench", dest="llb", action="store_true",
                    help="after each model's ladder, run llama-bench pp512/tg128 at 4K/32K/128K depth "
                         "with the same file and placement (server unloaded for it) - the community's number")
    ap.add_argument("--quick", action="store_true",
                    help="the short ladder: " + ",".join(map(str, QUICK_DEPTHS)))
    ap.add_argument("--dry-run", action="store_true", help="measure only, post nothing")
    # ⛔ ON BY DEFAULT. A refusal IS a result: the matrix needs it to tell a cell
    # PROVEN not to load from one nobody has tried, and an agent that cannot see
    # one will keep recommending a config the box has already rejected. It was
    # opt-in, so a run could hit a real cudaMalloc OOM, print an HTTP 500 and
    # record nothing - which happened twice on 2026-08-28 before anyone noticed
    # the flag was simply absent from the command.
    # Safe as a default because the evidence gate is strict: no cause line in
    # the box's own logs means INCONCLUSIVE and nothing is written.
    ap.add_argument("--no-record-refusals", dest="record_refusals",
                    action="store_false", default=True,
                    help="do NOT record an evidence-gated wont_fit when a load fails")
    ap.add_argument("--force", action="store_true", help="ignore an existing lock")
    args = ap.parse_args()
    if not TARGETS:
        sys.exit("no targets: copy targets.example.json to targets.json and describe your boxes")
    depths = (tuple(int(x) for x in args.depths.split(",")) if args.depths
              else QUICK_DEPTHS if args.quick else DEPTHS)

    raw_cmds = None
    if args.entries:
        raw_cmds = json.load(open(args.entries))
        if not args.target:
            sys.exit("--entries needs --target <box>")
        if args.models:
            raw_cmds = {m.strip(): raw_cmds[m.strip()] for m in args.models.split(",")}
        args.gap_fill = True
        args.models = ",".join(raw_cmds)
    if args.gap_fill:
        if not (args.target and args.models):
            sys.exit("--gap-fill needs --target <box> and --models <setup ids>")
        if TARGETS[args.target].get("engine") == "vllm":
            sys.exit(f"{args.target} is a vLLM box: no yaml to temp-deploy an entry on")
        plan = None
    elif args.all:
        plan = probe_fleet()
    elif args.target:
        served = served_models(TARGETS[args.target]["api"])
        models = ([m.strip() for m in args.models.split(",")] if args.models else served)
        unknown = [m for m in models if m not in served]
        if unknown:
            sys.exit(f"Not served on {args.target}: {unknown}\nServed: {served}")
        plan = {args.target: models}
    elif sys.stdin.isatty():
        plan = interactive_pick(probe_fleet())
    else:
        sys.exit("Non-interactive: pass --all, or --target [--models]. "
                 "(An agent kicking this off must be explicit.)")
    if plan is not None and not any(plan.values()):
        sys.exit("Nothing selected.")

    here = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(here, "results"), exist_ok=True)
    # ONE RUN PER BOX, not one run in total. The lock is per target, so two
    # boxes can be benched at the same time and a full sweep takes half the wall
    # clock.
    #
    # It used to be a single global lock, on the contract's "one box at a time"
    # ethic - which turned out to be folklore for INDEPENDENT boxes. Measured
    # 2026-08-28 on pris, qwen3.8-27b q4/f16, deep gen: 143.04 tok/s alone
    # vs 143.64 while cheyenne was mid-run (+0.4%, inside run-to-run noise;
    # prefill 3,032 vs 2,969). Nothing physical is shared and speeds come from
    # each ENGINE's own telemetry, not otago's clock, so a busy coordinator
    # cannot distort a number.
    #
    # ⛔ What the ethic DOES still forbid stands: never two runs against the SAME
    # box (they compete for its GPU), and the coordinator is never a target.
    locks = [os.path.join(here, "results", f".lock-{t}") for t in (plan or {args.target: []})]
    for lk in locks:
        if os.path.exists(lk) and not args.force:
            sys.exit(f"A run against {os.path.basename(lk)[6:]} appears active ({lk}). "
                     "--force only if stale. Other boxes can be benched concurrently.")
    for lk in locks:
        open(lk, "w").write(f"{os.path.basename(lk)[6:]} {os.getpid()}\n")

    mcp, ids, failed = None, {}, []
    if not args.dry_run:
        mcp = JRMcp()
        ids = mcp.model_ids()

    def submit(r):
        if mcp is None:
            return
        # A benchmark names the MODEL and carries its own setup. No catalog
        # lookup: an entry is optional by contract, so a model we have not
        # catalogued is still measurable and still posts (JR flags the row and
        # models_link_benchmarks attaches it later).
        cfg = json.loads(r["configJson"] or "{}")
        try:
            reply = mcp.call("models_record_benchmark", {
                "modelName": catalog_name(r["model"]),
                "metric": r["metric"], "value": r["value"],
                "machine": r["machine"], "scenario": r["scenario"],
                # The setup as first-class fields. runtime_facts (the running
                # server's /props) wins over anything read from the yaml.
                "quant": cfg.get("quant"),
                "contextTokens": cfg.get("ctx"),
                "slots": cfg.get("parallel"),
                "kvPrecision": cfg.get("cache_k") or cfg.get("cache_v"),
                "mtp": bool(cfg.get("spec_type")) and cfg.get("spec_type") != "none",   # any speculative decoding (JR lab_guide)
                # the server's thinking default (JR 0.9.0.201): TensorFold says it in its argv; vLLM's
                # launch kwargs, else a measured probe (thinking_probe); llama-server rows say nothing
                # until a flag proves it. Never assumed.
                "thinking": cfg.get("thinking_default"),
                "vision": cfg.get("vision"),
                "engine": cfg.get("engine"),
                "engineName": cfg.get("engine_name"),
                "engineVersion": cfg.get("engine_version"),
                "sourceRepo": cfg.get("source_repo"),
                "weightsGb": cfg.get("weights_gb"),
                # Tool-specific spillover only - the contract is above. THE
                # PLACEMENT FACTS GO TOO (2026-09-24, Jon: a run must carry every
                # differentiator): where the weights lived decides the number,
                # and tensor_split / load_mode / -ot were captured and then
                # dropped right here, so no row could say how it was split.
                "configJson": json.dumps({k: v for k, v in cfg.items() if k in
                                          ("llama_swap_name", "n_cpu_moe", "n_layers_total",
                                           "load_mode", "tensor_split", "tensor_overrides", "ubatch",
                                           "cache_v", "spec_type", "suite", "suite_name", "suite_version",
                                           "tensor_parallel", "spec_tokens", "gpu_memory_utilization",
                                           "ple_offload", "engram_offload", "thinking_default")}),
                "source": r["source"],
                # the tin behind a role-named machine (JR 0.9.0.196, 2026-10-02)
                "hardware": next((t.get("hardware") for t in TARGETS.values() if t["machine"] == r["machine"]), None),
                "notes": r["notes"], "capturedAt": r["capturedAt"]})
            # Posted only when JR says so: it can decline inside a successful MCP response.
            if not (reply or "").strip().startswith("Recorded"):
                raise RuntimeError((reply or "(empty reply)").strip()[:200])
        except Exception as e:
            failed.append(r)
            print(f"  !! submit failed ({e}) - row kept in outbox")

    if args.gap_fill:
        try:
            gap_fill(args.target, [m.strip() for m in args.models.split(",")],
                     args.parallel, mcp, ids, submit, depths, raw_cmds, args.llb)
            if failed:
                print(f"\n⚠ {len(failed)} row(s) failed to post.")
        finally:
            for lk in locks:
                if os.path.exists(lk):
                    os.remove(lk)
        return

    try:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for tname, models in plan.items():   # strictly sequential across boxes
            t = TARGETS[tname]
            print(f"\n=== {t['machine']} ({t['api']}) - {len(models)} model(s)"
                  + (", posting live over MCP" if mcp else ", dry run") + " ===")
            all_rows = []
            for m in models:
                print(f"\n[{m}]")
                try:
                    all_rows.extend(bench_model(t["api"], m, t["machine"],
                                                args.parallel, t["ssh"], submit, t["os"], depths, args.llb))
                except SystemExit:
                    raise
                except Exception as e:
                    if t.get("engine") == "vllm":
                        # No llama-swap to settle and no load to refuse: the server
                        # was already up, so a failure here is not a fit verdict.
                        print(f"  FAILED: {e} - no rows recorded for this model")
                    elif args.record_refusals:
                        wait_settled(t["api"])
                        try:
                            all_rows.extend(bench_model(t["api"], m, t["machine"],
                                                        args.parallel, t["ssh"], submit, t["os"], depths, args.llb))
                            continue
                        except SystemExit:
                            raise
                        except Exception as e2:
                            record_refusal(t, m, e2, mcp, ids)
                    else:
                        print(f"  FAILED: {e} - no rows recorded for this model")
            out = os.path.join(here, "results", f"{tname}_{stamp}.json")
            with open(out, "w") as f:
                json.dump({"target": tname, "machine": t["machine"], "rows": all_rows,
                           "unsubmitted": [r for r in failed if r["machine"] == t["machine"]]},
                          f, indent=2)
            print(f"\n{t['machine']}: {len(all_rows)} row(s) -> {out}")
        if failed:
            print(f"\n⚠ {len(failed)} row(s) failed to post - they are in the outbox files.")
    finally:
        for lk in locks:
            if os.path.exists(lk):
                os.remove(lk)


if __name__ == "__main__":
    main()
