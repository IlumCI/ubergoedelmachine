//! Verified self-training export — the STaR/ReST data engine.
//!
//!     DATASET=train.jsonl cargo run -p samaritan-run --example selftrain_export
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
//!   SAMARITAN_URL   model server (serve.ps1 -Role solver).
//!   DATASET         a *trainable* reasoning JSONL (NuminaMath, GSM-Symbolic
//!                   main/p1 — NOT the held-out set you measure on).
//!   LIMIT           cap items (default 200).
//!   OUT             output SFT JSONL (default reasoning-selftrain.jsonl).
//!   MAX_TOKENS, SEED  sampling controls.

use std::time::Duration;

use samaritan_agent::{Agent, AgentConfig, Constrain};
use samaritan_corpus::{load_reasoning, Split};
use samaritan_episode::{run_reasoning_episode, EpisodeConfig, REASONING_SYSTEM};
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
    let out = std::env::var("OUT").unwrap_or_else(|_| "reasoning-selftrain.jsonl".into());
    let max_tokens: u32 =
        std::env::var("MAX_TOKENS").ok().and_then(|s| s.parse().ok()).unwrap_or(2048);
    let seed = std::env::var("SEED").ok().and_then(|s| s.parse().ok()).unwrap_or(1);

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
        // Ollama tags models (name:latest) and matches exactly on its OpenAI
        // endpoint — set SAMARITAN_MODEL=samaritan-playout:latest for that backend.
        model: std::env::var("SAMARITAN_MODEL").unwrap_or_else(|_| "samaritan-playout".into()),
        temperature: 0.7, // some diversity: different attempts solve different items
        max_tokens,
        repeat_penalty: 1.1,
        constrain: Constrain::None,
        timeout: Duration::from_secs(900),
        max_retries: 1,
        seed: Some(seed),
        ..Default::default()
    });

    println!("server:  {base_url}");
    println!("dataset: {dataset} ({} items, attempting {})\n", corpus.tasks.len(), tasks.len());

    let mut ledger = Ledger::in_memory(Box::new(FixedClock("2026-09-13T00:00:00Z".into()))).unwrap();
    let cfg = EpisodeConfig::default();

    let mut rows: Vec<String> = Vec::new();
    let mut solved = 0usize;
    for (i, t) in tasks.iter().enumerate() {
        let out = match run_reasoning_episode(t, &agent, &mut ledger, &cfg) {
            Ok(o) => o,
            Err(e) => {
                eprintln!("  {:>3} error: {e}", i + 1);
                continue;
            }
        };
        if out.ending.solved() {
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
                rows.push(row.to_string());
            }
        }
        if (i + 1) % 10 == 0 {
            println!("  {}/{} attempted, {} kept", i + 1, tasks.len(), rows.len());
        }
    }

    std::fs::write(&out, rows.join("\n") + "\n").expect("write sft jsonl");

    let n = tasks.len().max(1);
    println!("\n── self-training set ───────────────────");
    println!("attempted: {}", tasks.len());
    println!("solved:    {} ({:.1}%)", solved, 100.0 * solved as f64 / n as f64);
    println!("kept:      {} verified traces", rows.len());
    println!("wrote {} rows to {out}", rows.len());
    if rows.is_empty() {
        println!("\nNothing verified — no signal to train on. Use an easier training split\n(GSM-Symbolic main, or GSM8K) so the solver can actually land some.");
    } else {
        println!("\nnext (on a cloud GPU — Kaggle T4 or Colab): train on this set,\n  python training/qdora_deviant.py {out} --allow-small\nthen convert the adapter to GGUF and serve it (training/README.md).");
    }
}
