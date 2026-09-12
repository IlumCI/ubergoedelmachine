//! Run the live solver over a reasoning corpus and report accuracy + calibration.
//!
//!     cargo run -p samaritan-run --example reason_eval
//!
//! The first real read on whether the substrate reasons, not just runs. It loads
//! a reasoning JSONL, has the local model answer each item through the same
//! run_reasoning_episode the search uses, grades against the key (the harness
//! grades, never the model), and reports solved-fraction, mean confidence, and
//! Brier calibration — plus a per-domain breakdown, because a system that only
//! wins one domain is not reasoning, it is remembering.
//!
//! Env:
//!   SAMARITAN_URL   model server (default http://127.0.0.1:8080/v1) — run
//!                   `serve.ps1 -Role solver` first.
//!   DATASET         reasoning JSONL to evaluate. Omit to use the trivial
//!                   built-in smoke set (a pipeline check, NOT a real eval).
//!   LIMIT           cap the number of items (default 20; the model is slow).
//!   SEED, MAX_TOKENS  sampling controls.
//!
//! The built-in questions are deliberately easy — they prove the pipeline end to
//! end (prompt -> think -> answer -> grade -> score). A real signal needs a real
//! corpus (a NuminaMath/GPQA slice, an HLE held-out subset); those are
//! operator-supplied, like the knowledge base and the eval grader.

use std::time::Duration;

use samaritan_agent::{Agent, AgentConfig, Constrain};
use samaritan_corpus::{load_reasoning, Split};
use samaritan_episode::{run_reasoning_episode, EpisodeConfig};
use samaritan_ledger::{FixedClock, Ledger};

/// A trivial, verifiable smoke set across a few domains. Not an evaluation — a
/// proof that the wiring works.
const SMOKE: &str = r#"
{"question":"What is 6 times 7? Give just the number.","answer":"42","domain":"math"}
{"question":"What is 2 to the power of 10? Give just the number.","answer":"1024","domain":"math"}
{"question":"What is the chemical symbol for gold?","answer":"Au","domain":"science"}
{"question":"What is the capital of France?","answer":"Paris","domain":"geography"}
{"question":"How many prime numbers are strictly between 10 and 20? Give just the number.","answer":"4","domain":"math"}
{"question":"If all Bloops are Razzies and all Razzies are Lazzies, are all Bloops Lazzies? Answer yes or no.","answer":"yes","domain":"logic"}
"#;

fn main() {
    let base_url =
        std::env::var("SAMARITAN_URL").unwrap_or_else(|_| "http://127.0.0.1:8080/v1".into());
    let limit: usize = std::env::var("LIMIT").ok().and_then(|s| s.parse().ok()).unwrap_or(20);
    let seed = std::env::var("SEED").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    let max_tokens: u32 =
        std::env::var("MAX_TOKENS").ok().and_then(|s| s.parse().ok()).unwrap_or(2048);

    let (text, source) = match std::env::var("DATASET") {
        Ok(path) => match std::fs::read_to_string(&path) {
            Ok(t) => (t, path),
            Err(e) => {
                eprintln!("could not read {path}: {e}");
                std::process::exit(1);
            }
        },
        Err(_) => (SMOKE.to_string(), "built-in smoke set".to_string()),
    };

    let corpus = match load_reasoning(&text, "eval", false, Split::HeldOut) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("could not load reasoning corpus: {e}");
            std::process::exit(1);
        }
    };
    let tasks: Vec<_> = corpus.tasks.iter().take(limit).collect();

    let agent = Agent::new(AgentConfig {
        base_url: base_url.clone(),
        model: "samaritan-playout".into(),
        // A little exploration helps a thinking model; not greedy, not wild.
        temperature: 0.6,
        // Room for a real thinking trace within the time budget.
        max_tokens,
        repeat_penalty: 1.1,
        constrain: Constrain::None,
        timeout: Duration::from_secs(900),
        max_retries: 1,
        seed: Some(seed),
        ..Default::default()
    });

    println!("server:  {base_url}");
    println!("dataset: {source} ({} items, showing {})\n", corpus.tasks.len(), tasks.len());

    let mut ledger = Ledger::in_memory(Box::new(FixedClock("2026-09-13T00:00:00Z".into()))).unwrap();
    let cfg = EpisodeConfig::default();

    let mut solved = 0usize;
    let mut conf_sum = 0.0;
    let mut brier_sum = 0.0;
    // domain -> (correct, total)
    let mut by_domain: std::collections::BTreeMap<String, (usize, usize)> = Default::default();

    for (i, t) in tasks.iter().enumerate() {
        let out = match run_reasoning_episode(t, &agent, &mut ledger, &cfg) {
            Ok(o) => o,
            Err(e) => {
                println!("  q{:>2} [{}]  ERROR: {e}", i + 1, t.domain());
                continue;
            }
        };
        let correct = out.ending.solved();
        let (conf, _) = out.predictions.first().copied().unwrap_or((0.5, false));
        if correct {
            solved += 1;
        }
        conf_sum += conf;
        let o = if correct { 1.0 } else { 0.0 };
        brier_sum += (conf - o).powi(2);
        let e = by_domain.entry(t.domain().to_string()).or_insert((0, 0));
        e.1 += 1;
        if correct {
            e.0 += 1;
        }
        println!(
            "  q{:>2} [{:<9}] {}  conf {:.2}  ({} tok)",
            i + 1,
            t.domain(),
            if correct { "OK  " } else { "MISS" },
            conf,
            out.tokens,
        );
        // On a miss, show what the model actually answered — the fastest way to
        // tell a real reasoning error from an extraction/grading edge.
        if !correct {
            if let Some(a) = &out.answer {
                let a = a.replace('\n', " ");
                let a = if a.len() > 100 { format!("{}…", &a[..100]) } else { a };
                println!("        got: {a:?}");
            }
        }
    }

    let n = tasks.len().max(1);
    println!("\n── result ──────────────────────────────");
    println!("accuracy:        {}/{} = {:.1}%", solved, tasks.len(), 100.0 * solved as f64 / n as f64);
    println!("mean confidence: {:.2}", conf_sum / n as f64);
    println!("Brier (calib):   {:.3}   (lower is better; 0.25 = always 0.5)", brier_sum / n as f64);
    println!("by domain:");
    for (d, (c, tot)) in &by_domain {
        println!("  {:<12} {}/{}", d, c, tot);
    }
    if source == "built-in smoke set" {
        println!("\n(This is the trivial smoke set — a wiring check. Point DATASET at a real\nreasoning corpus, and an HLE held-out subset, for a signal that means something.)");
    }
}
