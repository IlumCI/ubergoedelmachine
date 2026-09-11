# Samaritan

A self-improving agent harness for local dev/repo work, built around a nested
Monte Carlo search whose nesting axis is *depth of self-reference*, with every
self-modification gated by an anytime-valid statistical certificate and every
consequential action gated by a human.

See [`docs/design.md`](docs/design.md) for the architecture, the algorithm, and
how it is positioned against the Gödel-machine literature.

## Status

**The full harness is assembled.** Fourteen crates, 300 tests green. A run
composes corpus -> episode -> reflect -> search -> certify -> ledger into one
of three experimental arms (Solo / Critic / Adversarial), gated by the
anytime-valid certificate and bounded by a compute budget. The only thing
left is pointing it at a local model — which waits on cooling.

Measured, not asserted: the certificate's false-certification rate is **0.022**
against a bound of 0.05, and level-2 nested search reaches a known optimum in
400 evaluations where flat search and random search both stall at 5/8 given
more.

Inference setup — model, hardware arithmetic, and why llama.cpp rather than
RustLMHub — is in [docs/inference.md](docs/inference.md).

| Crate | State |
|---|---|
| `samaritan-dsl` | decision language, input provenance, mutation grammar, NRPA policy — done |
| `samaritan-kernel` | frozen core: routing, admission, utility, compute budget — done |
| `samaritan-ledger` | hash-chained log, calibration, containment, experiment + frontier metrics — done |
| `samaritan-adversary` | **The Deviant.** The Warden inverted: attacks the real guards, arena scores by novel landings — done |
| `samaritan-corpus` | history miner, sealed sandboxes, flake screening, splits, frontier set — done |
| `samaritan-exec` | confined actions, container isolation, timeouts, oracle runner — done |
| `samaritan-router` | tier dispatch, approval gating, misgrade detection, autonomy streaks — done |
| `samaritan-agent` | OpenAI-compatible client, GBNF-constrained output, seeded runs — done |
| `samaritan-cli` | inline approval prompt; no standing-permission option — done |
| `samaritan-episode` | level-0 rollout: sandbox, decide/route/execute loop, oracle, score — done |
| `samaritan-cert` | test martingale, α-investing, provenance check — done |
| `samaritan-search` | NRPA over self-reference depth, policy state, applier — done |
| `samaritan-reflect` | lessons mined from ledger outcomes (not self-report), fed to the search — done |
| `samaritan-run` | the runner: one full self-improvement run as one experimental arm — done |

## Building on Windows

The toolchain needs a working `dlltool` and assembler. A Chocolatey-installed
`windows-gnu` rustc ships an incomplete binutils set: `dlltool.exe` is present
under `<sysroot>/lib/rustlib/x86_64-pc-windows-gnu/bin/self-contained/` but
cannot spawn an assembler, so anything depending on `getrandom` (via `uuid`,
later `rand`) fails to compile with:

```
error: dlltool could not create import library ... CreateProcess
```

Fix, from an **elevated** shell:

```bash
choco install mingw -y
```

This deploys to `C:\ProgramData\mingw64\mingw64\bin` and adds it to the
machine PATH. Shells started before the install keep the old PATH, so either
open a new one or prepend it:

```bash
export PATH="/c/ProgramData/mingw64/mingw64/bin:$PATH"
```

`cargo check` works without any of this; `cargo build` and `cargo test` do not.

## Layout

```
crates/
  samaritan-dsl/      what the model may say, and how it may change itself
  samaritan-kernel/   what it may do, and what it may never change
  samaritan-ledger/   what it did, in a form it cannot retract
  samaritan-corpus/   tasks mined from history, sealed from their answers
  samaritan-exec/     doing things in a box, and reporting what was really done
  samaritan-router/   deciding what happens, asking when it must
  samaritan-agent/    asking the local model, and not trusting its answer
  samaritan-cli/      asking the human, and never answering for them
  samaritan-episode/  one task, start to score
  samaritan-search/   nested rollout policy adaptation over meta-levels
  samaritan-cert/     whether a self-modification may be kept
  samaritan-adversary/ the Deviant, and the arena it is caged in
  samaritan-reflect/  lessons mined from outcomes, never self-reported
  samaritan-run/      the whole thing, one run, one arm
scripts/              fetch the weights, tune the split, serve
config/samaritan.toml local model endpoint + arena settings
docs/design.md        the design of record
```

`crates/samaritan-kernel/` and `crates/samaritan-cert/` are listed in
`FROZEN_PATHS`. The harness cannot patch them, and an attempt to is logged as a
breach rather than scored as a failed experiment.
