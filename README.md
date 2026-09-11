# Samaritan

A self-improving agent harness for local dev/repo work, built around a nested
Monte Carlo search whose nesting axis is *depth of self-reference*, with every
self-modification gated by an anytime-valid statistical certificate and every
consequential action gated by a human.

See [`docs/design.md`](docs/design.md) for the architecture, the algorithm, and
how it is positioned against the Gödel-machine literature.

## Status

M0 complete. M1 in progress: 168 tests green across six crates.

| Crate | State |
|---|---|
| `samaritan-dsl` | decision language, input provenance, mutation grammar, NRPA policy — done |
| `samaritan-kernel` | frozen core: routing, admission, utility, compute budget — done |
| `samaritan-ledger` | hash-chained log, calibration, containment, experiment + frontier metrics — done |
| `samaritan-adversary` | **The Deviant.** Unleashed coevolving adversary; arena only — not started |
| `samaritan-corpus` | history miner, sealed sandboxes, flake screening, splits, frontier set — done |
| `samaritan-exec` | confined actions, process timeouts, real oracle runner — done |
| `samaritan-router` | tier dispatch, approval gating, misgrade detection, autonomy streaks — done |
| `samaritan-search`, `-cert`, `-agent`, `-cli` | not started |

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
config/samaritan.toml local model endpoint + arena settings
docs/design.md        the design of record
```

`crates/samaritan-kernel/` and `crates/samaritan-cert/` are listed in
`FROZEN_PATHS`. The harness cannot patch them, and an attempt to is logged as a
breach rather than scored as a failed experiment.
