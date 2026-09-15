//! Verified self-training export — the STaR/ReST data engine.
//!
//!     DATASET=trainable-blend.jsonl cargo run -p samaritan-run --example selftrain_export
//!
//! The local half of "modify the model": run the solver over a *training*
//! reasoning corpus, keep only the traces whose answer the harness graded
//! **correct**, and write them as fine-tuning data. The model then learns from
//! its own verified reasoning — the classic self-improvement recipe (STaR:
//! Zelikman et al.; ReST) — and the fine-tune itself runs on a cloud GPU (see
//! training/README.md), since a 4 GB laptop serves but cannot train.
//!
//! Two disciplines carried from the rest of the system:
//!   - The harness grades, never the model. Only verified-correct traces are
//!     kept, so the training signal cannot be self-flattering.
//!   - The rows are the exact `{system, user, completion, verdict, reward}`
//!     shape `training/qdora_deviant.py` already reads, so the reasoning
//!     self-training set trains through the same, unchanged trainer.
//!
//! Env:
//!   SAMARITAN_URL     model server (serve.ps1 -Role solver, or the A100 tunnel).
//!   SAMARITAN_API_KEY / SAMARITAN_MODEL  for a remote endpoint (Ollama wants
//!                     the tag, e.g. samaritan-playout:latest).
//!   DATASET           a *trainable* reasoning JSONL (the multi-domain blend
//!                     from fetch-reasoning-set.ps1 -Dataset blend — NOT the
//!                     held-out set you measure on).
//!   LIMIT             cap items (default 200).
//!   OUT               output SFT JSONL (default reasoning-selftrain.jsonl),
//!                     appended per item so a crash loses nothing.
//!   USE_TOOLS         1/true to let the teacher run Python mid-reasoning — the
//!                     tool-grounded traces are the ones worth distilling.
//!   MAX_TOOL_STEPS / PYTHON_BIN / TOOL_TIMEOUT_SECS  tool-loop knobs.
//!   MAX_TOKENS        per-item budget (default 32768 — a reasoning floor).
//!   HTTP_TIMEOUT_SECS request timeout override; default scales with MAX_TOKENS
//!                     (a big budget needs wall-clock, or it dies as a transport
//!                     timeout — the q14 lesson).
//!   SELFTRAIN_RESUME  progress JSONL; defaults to <OUT>.progress.jsonl so a
//!                     dropped runtime costs only the unfinished items. Graded
//!                     items (solved or missed) are skipped on rerun; transport
//!                     failures are NOT recorded, so they retry. Set it to ""
//!                     to disable; delete OUT and the progress file for a fresh
//!                     run. Deliberately NOT named RESUME: reason_eval uses that,
//!                     and a shell that ran both leaked this export's progress
//!                     into the eval's file.
//!   SEED              sampling seed.

use std::io::Write as _;
use std::time::Duration;

use samaritan_agent::{Agent, AgentConfig, Constrain};
use samaritan_corpus::{load_reasoning, Split};
use samaritan_episode::{
    run_reasoning_episode, Ending, EpisodeConfig, ReasoningSolver, ToolLoopSolver,
    REASONING_SYSTEM,
};
use samaritan_ledger::{FixedClock, Ledger};
use serde_json::json;

fn main() {
    let base_url =
        std::env::var("SAMARITAN_URL").unwrap_or_else(|_| "http://127.0.0.1:8080/v1".into());
    let dataset = std::env::var("DATASET").unwrap_or_else(|_| {
        eprintln!("set DATASET to a trainable reasoning JSONL (not the held-out eval set)");
        std::process::exit(2);
    });
    let limit: usize = std::env::var("LIMIT").ok().and_then(|s| s.parse().ok()).unwrap_or(200);
    let out_path = std::env::var("OUT").unwrap_or_else(|_| "reasoning-selftrain.jsonl".into());
    let max_tokens: u32 =
        std::env::var("MAX_TOKENS").ok().and_then(|s| s.parse().ok()).unwrap_or(32768);
    // The request timeout must track the token budget (same formula as
    // reason_eval): a big budget needs the wall-clock to spend it, or the
    // request aborts mid-generation as `transport: timeout: global`.
    let timeout_secs: u64 = std::env::var("HTTP_TIMEOUT_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| 900.max(u64::from(max_tokens) / 20 + 300));
    let seed = std::env::var("SEED").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    // The code-execution tool loop. For distillation this is not a nicety: the
    // whole point of the flywheel is that tool-grounded verified traces are the
    // training data, so the teacher should reason the way we want the student to.
    let use_tools = std::env::var("USE_TOOLS").map(|v| v == "1" || v == "true").unwrap_or(false);
    let max_tool_steps: u32 =
        std::env::var("MAX_TOOL_STEPS").ok().and_then(|s| s.parse().ok()).unwrap_or(4);
    let python_bin = std::env::var("PYTHON_BIN").unwrap_or_else(|_| "python".into());
    let tool_timeout = Duration::from_secs(
        std::env::var("TOOL_TIMEOUT_SECS").ok().and_then(|s| s.parse().ok()).unwrap_or(60),
    );

    let text = std::fs::read_to_string(&dataset).unwrap_or_else(|e| {
        eprintln!("could not read {dataset}: {e}");
        std::process::exit(1);
    });
    // Loaded as Train — this is data to learn from, and label matters here (it
    // must not be a held-out set).
    let corpus = load_reasoning(&text, "selftrain", true, Split::Train).unwrap_or_else(|e| {
        eprintln!("could not load corpus: {e}");
        std::process::exit(1);
    });
    let tasks: Vec<_> = corpus.tasks.iter().take(limit).collect();

    let agent = Agent::new(AgentConfig {
        base_url: base_url.clone(),
        // For a remote keyed solver (A100 via a tunnel); empty for the local one.
        api_key: std::env::var("SAMARITAN_API_KEY").unwrap_or_default(),
        model: std::env::var("SAMARITAN_MODEL").unwrap_or_else(|_| "samaritan-playout".into()),
        temperature: 0.7, // some diversity: different attempts solve different items
        max_tokens,
        repeat_penalty: 1.1,
        constrain: Constrain::None,
        timeout: Duration::from_secs(timeout_secs),
        max_retries: 1,
        seed: Some(seed),
        ..Default::default()
    });
    let work_dir = std::env::temp_dir().join("samaritan-toolloop");
    let _ = std::fs::create_dir_all(&work_dir);
    let tool_solver =
        ToolLoopSolver::new(&agent, max_tool_steps, python_bin.clone(), tool_timeout, work_dir);
    let solver: &dyn ReasoningSolver = if use_tools { &tool_solver } else { &agent };

    println!("server:  {base_url}");
    println!("dataset: {dataset} ({} items, attempting {})", corpus.tasks.len(), tasks.len());
    println!("budget:  {max_tokens} tok/item, {timeout_secs}s request timeout");
    if use_tools {
        println!("tools:   python loop, up to {max_tool_steps} steps, {}s each", tool_timeout.as_secs());
    }
    println!();

    // Crash-safety, on by default: this is a long paid run over a droppable
    // tunnel. Every *graded* attempt (solved or missed) is recorded so a rerun
    // skips it; kept traces are appended to OUT as they land. Transport failures
    // are not recorded, so they retry on the next run.
    // A DISTINCT env name from reason_eval's RESUME, on purpose. Both tools ran
    // in one PowerShell session where RESUME was still set from the eval, so this
    // export appended its progress into the eval's file and two schemas ended up
    // interleaved. Distinct names stop that at the source; the schema guard below
    // stops it even if a path is shared anyway.
    let resume_path = match std::env::var("SELFTRAIN_RESUME") {
        Ok(p) => {
            let p = p.trim().to_string();
            if p.is_empty() { None } else { Some(p) }
        }
        Err(_) => Some(format!("{out_path}.progress.jsonl")),
    };
    let mut done: std::collections::HashSet<String> = Default::default();
    if let Some(p) = &resume_path {
        if let Ok(txt) = std::fs::read_to_string(p) {
            for line in txt.lines().filter(|l| !l.trim().is_empty()) {
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
                    // Ours only: a foreign row (reason_eval writes `correct`,
                    // not `solved`) is ignored rather than silently misread.
                    if v.get("solved").is_some() {
                        if let Some(q) = v["question"].as_str() {
                            done.insert(q.to_string());
                        }
                    }
                }
            }
            if !done.is_empty() {
                println!("resume: {} item(s) already attempted per {p}\n", done.len());
            }
        }
    }
    let mut progress_file = resume_path.as_ref().map(|p| {
        std::fs::OpenOptions::new().create(true).append(true).open(p).expect("open RESUME file")
    });
    let mut out_file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&out_path)
        .expect("open OUT file");

    let mut ledger = Ledger::in_memory(Box::new(FixedClock("2026-09-13T00:00:00Z".into()))).unwrap();
    let cfg = EpisodeConfig::default();

    let mut kept = 0usize;
    let mut solved = 0usize;
    let mut attempted = 0usize;
    let mut failed = 0usize;
    let mut skipped = 0usize;
    for (i, t) in tasks.iter().enumerate() {
        if done.contains(&t.prompt) {
            skipped += 1;
            continue;
        }
        let out = match run_reasoning_episode(t, solver, &mut ledger, &cfg) {
            Ok(o) => o,
            Err(e) => {
                eprintln!("  {:>4} error: {e}", i + 1);
                continue;
            }
        };
        // A transport failure is not a graded attempt: surface it, keep it out
        // of the solve-rate, and let the next run retry it.
        if let Ending::AgentFailed { detail } = &out.ending {
            failed += 1;
            let d: String = detail.replace('\n', " ").chars().take(200).collect();
            println!("  {:>4} [{:<12}] FAIL  {d}", i + 1, t.domain());
            continue;
        }

        attempted += 1;
        let ok = out.ending.solved();
        if ok {
            solved += 1;
            // Keep the verified-correct trace as a training example. The model
            // learns to reproduce its own reasoning that actually worked.
            if let Some(trace) = out.trace {
                let row = json!({
                    "system": REASONING_SYSTEM,
                    "user": t.prompt,
                    "completion": trace,
                    "verdict": "solved",
                    "reward": 1.0,
                    "round": i,
                    "domain": t.domain(),
                });
                let _ = writeln!(out_file, "{row}");
                // fsync, not flush: a verified trace costs minutes of teacher
                // time, so it should be on the disk before we move on.
                let _ = out_file.sync_all();
                kept += 1;
            }
        }
        println!(
            "  {:>4} [{:<12}] {}  ({} tok)  kept {}",
            i + 1,
            t.domain(),
            if ok { "OK  " } else { "MISS" },
            out.tokens,
            kept,
        );
        if let Some(f) = progress_file.as_mut() {
            let rec = json!({ "question": t.prompt, "solved": ok });
            let _ = writeln!(f, "{rec}");
            let _ = f.sync_all();
        }
    }

    let n = attempted.max(1);
    println!("\n── self-training set ───────────────────");
    if skipped > 0 {
        println!("skipped:   {skipped} already attempted (resume)");
    }
    println!("attempted: {attempted}");
    println!("solved:    {} ({:.1}%)", solved, 100.0 * solved as f64 / n as f64);
    if failed > 0 {
        println!("failed:    {failed} transport errors (not graded; rerun retries them)");
    }
    println!("kept:      {kept} verified traces this run -> {out_path}");
    if kept == 0 && attempted > 0 {
        println!("\nNothing verified this run — no new signal. If the solve-rate is near zero,\npoint at an easier slice so the teacher can actually land some.");
    } else {
        println!("\nnext (on a cloud GPU — Colab G4/H100 + high-RAM): train the student on it,\n  python training/qdora_deviant.py {out_path} --base-model Qwen/Qwen3-4B-Thinking-2507 --output adapters/reasoning-qdora\nthen merge, convert to GGUF, and serve it (training/README.md).");
    }
}
