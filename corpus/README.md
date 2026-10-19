# The corpus

`kubernetes/` is a FROZEN snapshot of three sections of the Kubernetes concepts docs
(github.com/kubernetes/website, `content/en/docs/concepts/{architecture,configuration,containers}`,
Markdown only) at commit ddaf814, 2026-10-19: 22 files, 228,451 chars, which GLM-5.3-Flash's
tokenizer reads as 51,417 tokens (measured 2026-10-18; the calibration `bench.py` cuts rungs with).

The Kubernetes documentation is © The Kubernetes Authors, licensed CC BY 4.0
(https://creativecommons.org/licenses/by/4.0/). Files are unchanged.

Every depth rung is this corpus in sorted path order, cut to the rung; rungs past 51.4K
cycle it with a pass marker. The served `prompt_tokens` rides each row's notes.

⛔ NEVER refresh it in place. A changed corpus is a changed scenario: re-snapshot only with a
`SUITE_VERSION` bump in bench.py, and say so in the version log there.
