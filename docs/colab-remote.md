# The A100 as Samaritan's remote solver (and trainer), over SSH

The p2 eval proved the local 4B is the ceiling on hard reasoning. A Colab Pro+
A100 (40 GB) can serve a *much* stronger reasoning model — and because the
harness talks to any OpenAI-compatible URL, using it needs **no code change**,
only an `SSH -L` port-forward and a `SAMARITAN_URL`.

This keeps the original cheap+strong split the design always wanted: the local
4B does the thousands of cheap level-0 playouts; the A100 model takes the hard
reasoning calls.

**What this is and isn't.** Good for eval pushes, self-training data generation,
and fine-tunes — anything that fits a session. Colab Pro+ background runs cap at
~24 h and can disconnect, so this is **not** a 24/7 deployment; a persistent
solver would want a rented VPS later. And consumer Colab has no native sshd:
`colab-ssh` and friends stand one up behind a **cloudflared** tunnel, so "SSH"
here is SSH-over-cloudflared. The port-forward pattern is the same whatever tool
you use.

---

## Part 1 — SSH into the runtime

In a Colab cell (A100 runtime selected):

```python
# One-time: a tunnel + sshd on the runtime. Set a password or drop your pubkey.
!pip -q install colab_ssh --upgrade
from colab_ssh import launch_ssh_cloudflared
launch_ssh_cloudflared(password="choose-a-strong-one")
```

It prints an SSH `ProxyCommand` / host to use locally (cloudflared handles the
transport). Add it to your `~/.ssh/config` as host `colab` so the commands below
work. Test from the laptop: `ssh colab "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"` → should print the A100.

## Part 2 — serve a strong reasoning model on the A100

vLLM is the right server here: native OpenAI API, fast batched generation (which
the many parallel playouts want), and it fits a big model on 40 GB. In a Colab
cell:

```python
!pip -q install vllm
# Qwen3-14B-Thinking fits comfortably in fp16-ish on 40 GB; a 32B fits 4-bit.
# --served-model-name is the alias the harness asks for, so nothing changes.
!nohup python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-14B-Thinking-2507 \
    --served-model-name samaritan-playout \
    --port 8000 --max-model-len 8192 > vllm.log 2>&1 &
# wait for "Uvicorn running" in vllm.log, then it's serving on the runtime's :8000
```

Model choice, honestly: **Qwen3-14B-Thinking** is the sweet spot on a 40 GB
A100 — a large step up from the 4B on p2/GPQA, comfortable in memory, fast under
vLLM. A 32B (4-bit) fits and reasons better still, slower. Bench p2 with each and
keep whichever wins per unit time.

## Part 3 — forward the port and point the harness at it

From the laptop (a second terminal, left open):

```bash
ssh -N -L 8080:localhost:8000 colab      # forward local 8080 -> Colab vLLM :8000
```

Then the harness uses the A100 model with no change — its default URL already
points at local `:8080`:

```powershell
$env:DATASET = "$env:USERPROFILE\models\reasoning\gsm-symbolic-p2.jsonl"
cargo run -p samaritan-run --example reason_eval    # now answered by the 14B on the A100
```

Do **not** run `serve.ps1` at the same time — the forward already occupies
`:8080`. `serve.ps1 -Role solver` (the local 4B) is the fallback for when the
A100 session is gone, or the cheap playout model in a hybrid run.

## Part 4 — train on the same runtime

The self-training fine-tune runs on the A100 too. Generate the verified set
locally (it needs the solver — use either the local 4B or the forwarded A100),
copy it up, and train:

```bash
# local: produce verified traces (see training/README.md)
cargo run -p samaritan-run --example selftrain_export     # -> reasoning-selftrain.jsonl
scp reasoning-selftrain.jsonl colab:~/                     # up to the runtime
scp training/qdora_deviant.py training/requirements.txt colab:~/
ssh colab "pip -q install -r requirements.txt && python qdora_deviant.py reasoning-selftrain.jsonl --base-model Qwen/Qwen3-4B-Thinking-2507"
scp -r colab:~/adapters/deviant-qdora ./adapters/          # bring the adapter home
```

Then merge + convert to GGUF (`training/README.md`) and serve it locally, or
serve the merged model on the A100 via Part 2 and keep going.

## Security & housekeeping

- Use a strong password or, better, key-only auth for the tunnel; anyone with
  the cloudflared host + creds reaches your runtime.
- Kill the vLLM process and the tunnel when done; a forgotten A100 burns your
  Pro+ compute-units budget.
- Don't leave secrets in Colab cells or `vllm.log`.
