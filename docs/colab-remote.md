# The A100 as Samaritan's remote solver (and trainer)

The p2 eval proved the local 4B is the ceiling on hard reasoning. A Colab Pro+
A100 (40 GB) can serve a far stronger reasoning model, and the harness talks to
any OpenAI-compatible URL — so using it needs **no code change**, only a
`SAMARITAN_URL` (and, for a keyed endpoint, `SAMARITAN_API_KEY`, now supported by
the reasoning examples).

Keeps the cheap+strong split the design always wanted: the local 4B does the
thousands of cheap level-0 playouts; the A100 model takes the hard reasoning
calls.

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
pip -q install vllm
# Qwen3-14B-Thinking fits fp16-ish on 40 GB; a 32B fits 4-bit. --served-model-name
# is the alias the harness asks for, so nothing changes on our side. A key so the
# public tunnel URL is not open to the world.
KEY=$(python -c "import secrets;print(secrets.token_urlsafe(24))"); echo "API KEY: $KEY"
nohup python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-14B-Thinking-2507 \
    --served-model-name samaritan-playout \
    --api-key "$KEY" --port 8000 --max-model-len 8192 > vllm.log 2>&1 &
# wait for "Uvicorn running" in vllm.log, then publish the port:
wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared && chmod +x cloudflared
./cloudflared tunnel --url http://localhost:8000    # prints a https://<random>.trycloudflare.com URL
```

Model choice, honestly: **Qwen3-14B-Thinking** is the sweet spot on a 40 GB
A100 — a large step up from the 4B on p2/GPQA, comfortable, fast under vLLM. A 32B
(4-bit) reasons better still, slower. Bench p2 with each and keep whichever wins
per unit time.

## Point the harness at it (local, any OS)

```powershell
$env:SAMARITAN_URL = "https://<random>.trycloudflare.com/v1"   # the tunnel URL + /v1
$env:SAMARITAN_API_KEY = "<the KEY printed above>"
$env:DATASET = "$env:USERPROFILE\models\reasoning\gsm-symbolic-p2.jsonl"
cargo run -p samaritan-run --example reason_eval    # now answered by the 14B on the A100
```

No `serve.ps1` at the same time (that's the local 4B). This is the first real read
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

- The API key is what protects the public tunnel URL — don't paste it anywhere
  shared, and rotate it per session.
- `colab stop -s samaritan` (or kill the vLLM + cloudflared processes) when done;
  a forgotten A100 burns Pro+ compute units fast.
