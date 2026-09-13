# The A100 as Samaritan's remote solver (and trainer)

The p2 eval proved the local 4B is the ceiling on hard reasoning. A Colab Pro+
A100 (40 GB) can serve a far stronger reasoning model, and the harness talks to
any OpenAI-compatible URL — so using it needs **no code change**, only a
`SAMARITAN_URL` (and, for a keyed endpoint, `SAMARITAN_API_KEY`, now supported by
the reasoning examples).

Keeps the cheap+strong split the design always wanted: the local 4B does the
thousands of cheap level-0 playouts; the A100 model takes the hard reasoning
calls.

## The turnkey path (start here)

**[`docs/colab/samaritan_a100.ipynb`](colab/samaritan_a100.ipynb)** is a run-all
notebook that does the whole serve side: install Ollama, pull **Qwen3.8-27B
(Q8_0)**, alias it to `samaritan-playout`, open a cloudflared tunnel, and print the
exact `SAMARITAN_URL` line to paste locally. Open it in Colab (Runtime → A100), Run
all, wait for the tunnel URL. No WSL, no CLI. The sections below are the manual /
scriptable version of the same thing, plus the CLI route for people who want it.

## Read these constraints first

- **The official `google-colab-cli` is Linux/macOS only — not Windows.** On this
  Windows box, run it inside **WSL**. (Verified against the repo README.)
- **`colab ssh` has no documented port-forward** (`-L`/`-N`) — it's a WebSocket
  shell or a `--proxy-mode` ProxyCommand bridge. So don't reach the model through
  an SSH tunnel; expose its HTTP port with **cloudflared** instead (below).
- **Colab isn't 24/7** — Pro+ background runs cap ~24 h and can drop. Fine for
  eval pushes, self-training generation, and fine-tunes; not a deployment. A
  persistent solver would want a rented VPS later.
- Flags/behaviour below are from the repo README; anything it doesn't state,
  confirm with `colab <cmd> --help` on your install rather than trusting me.
- **Serve GGUF via Ollama, not FP8 via vLLM.** The A100 is Ampere (SM80) with no
  FP8 tensor cores, and vLLM has **no FP8-MoE support on Ampere** — an FP8 MoE
  checkpoint (e.g. Qwen3-30B-A3B-FP8) downloads ~30 GB and then *crashes at load*
  (vLLM issue #35922). Ollama runs the GGUF reliably, bundles its own CUDA, and
  needs no build. Dense FP8 would run on vLLM via Marlin; MoE FP8 does not.

## Provision the A100 (the CLI, in WSL)

```bash
pipx install google-colab-cli        # or: pip install google-colab-cli
colab --auth=oauth2 sessions         # first call triggers the copy-paste OAuth flow
colab new -s samaritan --gpu A100 --high-mem     # Pro+ entitlement; A100 shape
colab status -s samaritan            # confirm the A100 is up
colab ssh -s samaritan               # a shell on the runtime
#   colab sessions / colab stop -s samaritan   # list / tear down (stop when done!)
```

## Serve a strong reasoning model + expose it

In the `colab ssh` shell (or a notebook cell) on the runtime:

```bash
# Ollama bundles its own CUDA — no build against Colab's stack. Its installer
# extracts a zstd tarball and Colab lacks zstd, so install that first. Wipe any
# partial prior install so the llama-server runner lands too (a partial extract
# serves /v1 but 500s at generation with "llama-server binary not found"), then
# confirm the runner is present. Colab has no systemd, so start the daemon by hand.
apt-get -qq install -y zstd || (apt-get -qq update && apt-get -qq install -y zstd)
rm -rf /usr/local/lib/ollama
curl -fsSL https://ollama.com/install.sh | sh
ls /usr/local/lib/ollama/llama-server   # must exist; if not, the install is broken
# Flash attention + an 8-bit KV cache roughly halve the context cache (the quant
# only takes effect with FA on); together they're what lets a 128k window fit next
# to the ~30 GB model on 40 GB.
OLLAMA_KEEP_ALIVE=-1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 nohup ollama serve > ollama.log 2>&1 &
# Qwen3.8-27B (Q8_0, ~30 GB) is the strong reasoning model that fits 40 GB. The
# -mtp- build adds the multi-token-prediction draft head (self-speculative decode,
# faster gen at equal quality). Copy it to samaritan-playout (the alias the harness
# asks for) with a 128k context baked in. num_ctx is load-bearing: past it Ollama
# does NOT error, it silently drops the oldest tokens, so a long derivation forgets
# its own early work and loops — the real cause of the answer-less truncations we
# saw at the old 16k. The hybrid attention (only 16 of 64 layers keep a growing KV
# cache) + the q8 cache hold 128k to ~4 GB. (262144 is the model's max, but at ~8 GB
# of cache it sits at the 40 GB edge — 131072 is the safe floor.)
ollama pull qwen3.8:27b-mtp-q8_0
printf 'FROM qwen3.8:27b-mtp-q8_0\nPARAMETER num_ctx 131072\n' > Modelfile
ollama create samaritan-playout -f Modelfile
# publish port 11434. --http-host-header is required: Ollama returns 403 for any
# Host but localhost (DNS-rebinding guard), so rewrite it before the origin.
wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared && chmod +x cloudflared
./cloudflared tunnel --url http://localhost:11434 --http-host-header localhost:11434   # prints a https://<random>.trycloudflare.com URL
```

Model choice, honestly: **Qwen3.8-27B (Q8_0 GGUF)** is the sweet spot on a 40 GB
A100 — a large step up from the 4B on p2/GPQA, and Q8_0 (~30 GB) fits with room for
context. The `-mtp-` build is the one to serve: same weights, plus a multi-token-
prediction draft head for self-speculative decoding (faster generation at equal
quality). There is no Q6_K in the Ollama library for the 27B and bf16 (56 GB) won't
fit, so Q8_0 is the tag; plain `qwen3.8:27b-q8_0` (no MTP) or `qwen3.8:27b`
(Q4_K_M, 18 GB) are fallbacks. We serve GGUF via Ollama rather than FP8 via vLLM
because vLLM has no FP8-MoE support on the A100 (see the constraints above). Serve
it with a **128k `num_ctx`** and flash attention + `q8_0` KV cache on: the window
must exceed the longest thinking trace or Ollama silently trims the front of the
derivation (it does not error), which is what stalled the hardest AIME items at the
old 16k. To read
real throughput, hit Ollama's native `/api/chat` (`stream:false`) and divide
`eval_count` by `eval_duration` — that's server-side, so it's clean decode speed.
Ollama has **no API-key auth**, so the random tunnel URL is the only guard — stop
the runtime when done.

## Point the harness at it (local, any OS)

```powershell
$env:SAMARITAN_URL = "https://<random>.trycloudflare.com/v1"   # the tunnel URL + /v1
$env:SAMARITAN_API_KEY = "ollama"                              # any value; Ollama ignores it
$env:SAMARITAN_MODEL = "samaritan-playout:latest"             # Ollama matches the tag exactly
$env:MAX_TOKENS = "32768"                                      # gen budget; reason_eval defaults to this now
$env:DATASET = "$env:USERPROFILE\models\reasoning\gsm-symbolic-p2.jsonl"
cargo run -p samaritan-run --example reason_eval    # now answered by Qwen3.8-27B on the A100
```

Qwen recommends temp 1.0 / top_p 0.95 / top_k 20 / repeat 1.0 for this model in
thinking mode; `reason_eval` currently sends 0.6 / 1.1 (tuned for the local 4B),
which is fine for a first read. No `serve.ps1` at the same time (that's the local
4B). This is the first real read
from a *capable* substrate — it should crush the 4B's 2/10 on p2.

## Train on the same runtime

Generate the verified set locally (`selftrain_export`, see `training/README.md`),
get it onto the runtime, and fine-tune there. The CLI's file transfer isn't
documented, so the reliable routes are HF or Google Drive:

```bash
# get the SFT set onto the runtime — e.g. push to a private HF repo locally, then
# in the colab ssh shell:
pip -q install -r requirements.txt        # after copying training/ up, or clone the repo
python qdora_deviant.py reasoning-selftrain.jsonl --base-model Qwen/Qwen3-4B-Thinking-2507
# bring the adapter home the same way (HF/Drive), then merge + convert to GGUF
```

Or, if you prefer no WSL/CLI at all: do the same steps in a **browser Colab
notebook** — `files.upload()` the SFT set (or mount Drive), `!pip install`,
`!python qdora_deviant.py …`, save the adapter to Drive. The notebook path is
fully cross-platform; the CLI is just the scriptable version of it.

## Housekeeping

- Ollama has no auth, so the random tunnel URL is the *only* thing guarding the
  endpoint — don't paste it anywhere shared, and it changes each session anyway.
- `colab stop -s samaritan` (or kill the `ollama serve` + cloudflared processes,
  or just stop the runtime) when done; a forgotten A100 burns Pro+ units fast.
