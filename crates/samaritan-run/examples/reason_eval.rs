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
//!   SAMARITAN_URL     model server (default http://127.0.0.1:8080/v1) — run
//!                     `serve.ps1 -Role solver`, or a remote one (docs/colab-remote.md).
//!   SAMARITAN_API_KEY / SAMARITAN_MODEL  key and served-model name for a remote
//!                     endpoint (Ollama needs the tag, e.g. samaritan-playout:latest).
//!   DATASET           reasoning JSONL to evaluate. Omit to use the trivial
//!                     built-in smoke set (a pipeline check, NOT a real eval).
//!   LIMIT             cap the number of items (default 20; the model is slow).
//!   RESUME            path to a progress JSONL: answered items are skipped and new
//!                     ones appended, so a dropped runtime costs only the unfinished
//!                     items. Rerun with the same RESUME (and new URL) to continue;
//!                     delete the file for a fresh run.
//!   MAX_TOKENS        per-item generation budget (default 32768 — a reasoning
//!                     floor). Independent of the server's context window; keep it
//!                     under num_ctx and a trace is never silently trimmed.
//!   SEED              sampling seed.
//!
//! The built-in questions are deliberately easy — they prove the pipeline end to
//! end (prompt -> think -> answer -> grade -> score). A real signal needs a real
//! corpus (a NuminaMath/GPQA slice, an HLE held-out subset); those are
//! operator-supplied, like the knowledge base and the eval grader.

use std::io::Write;
use std::time::Duration;

use samaritan_agent::{Agent, AgentConfig, Constrain};
use samaritan_corpus::{load_reasoning, Split};
use samaritan_episode::{run_reasoning_episode, EpisodeConfig, Ending, ReasoningSolver, ToolLoopSolver};
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
    // Generation budget per item. A reasoning floor, not the old 2048: a hard
    // problem needs room to actually finish its derivation. This is the *budget*;
    // the server's num_ctx is the *window* (128k in docs/colab-remote.md), and the
    // two are independent — a budget under the window never gets silently trimmed.
    let max_tokens: u32 =
        std::env::var("MAX_TOKENS").ok().and_then(|s| s.parse().ok()).unwrap_or(32768);
    // The client's hard wall on one request, and it MUST track the token budget.
    // At the A100's ~30-40 tok/s a 65k-token answer needs ~30 min, so a fixed 900s
    // timeout aborts a big-budget generation mid-thought and reports it as a
    // transport failure — the model was still working, it just outran the clock.
    // Default to a budget-derived ceiling (a conservative ~20 tok/s floor + margin);
    // HTTP_TIMEOUT_SECS overrides. The request returns as soon as the model stops,
    // so this only caps the worst case, it doesn't spend it.
    let timeout_secs: u64 = std::env::var("HTTP_TIMEOUT_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| 900.max(u64::from(max_tokens) / 20 + 300));
    // Optional code-execution tool loop. USE_TOOLS=1 turns it on: the model may run
    // Python mid-reasoning and self-verify before answering. MAX_TOOL_STEPS bounds
    // iterations, PYTHON_BIN picks the interpreter, TOOL_TIMEOUT_SECS caps one run.
    let use_tools = std::env::var("USE_TOOLS").map(|v| v == "1" || v == "true").unwrap_or(false);
    let max_tool_steps: u32 =
        std::env::var("MAX_TOOL_STEPS").ok().and_then(|s| s.parse().ok()).unwrap_or(4);
    let python_bin = std::env::var("PYTHON_BIN").unwrap_or_else(|_| "python".into());
    let tool_timeout = Duration::from_secs(
        std::env::var("TOOL_TIMEOUT_SECS").ok().and_then(|s| s.parse().ok()).unwrap_or(60),
    );

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
        // Set when the solver is a remote endpoint behind a key (e.g. vLLM on an
        // A100 exposed through a cloudflared tunnel — see docs/colab-remote.md).
        // Empty for the local server, which wants nothing.
        api_key: std::env::var("SAMARITAN_API_KEY").unwrap_or_default(),
        // The served model name. Ollama tags its models (`name:latest`), and its
        // OpenAI endpoint matches exactly — so against an Ollama backend set
        // SAMARITAN_MODEL=samaritan-playout:latest. Defaults to the bare alias the
        // local llama.cpp server uses.
        model: std::env::var("SAMARITAN_MODEL").unwrap_or_else(|_| "samaritan-playout".into()),
        // A little exploration helps a thinking model; not greedy, not wild.
        temperature: 0.6,
        // Room for a real thinking trace within the time budget.
        max_tokens,
        repeat_penalty: 1.1,
        constrain: Constrain::None,
        timeout: Duration::from_secs(timeout_secs),
        max_retries: 1,
        seed: Some(seed),
        ..Default::default()
    });

    // PathChecked-style: the tool loop shells `python` in a scratch dir on this
    // host (fast, trusted). The solver is chosen once here — the tool loop when
    // USE_TOOLS is set, else the single-shot Agent — and the episode path below is
    // identical for both.
    let work_dir = std::env::temp_dir().join("samaritan-toolloop");
    let _ = std::fs::create_dir_all(&work_dir);
    let tool_solver = use_tools.then(|| {
        ToolLoopSolver::new(&agent, max_tool_steps, python_bin.clone(), tool_timeout, work_dir.clone())
    });
    let solver: &dyn ReasoningSolver = match &tool_solver {
        Some(s) => s,
        None => &agent,
    };

    println!("server:  {base_url}");
    println!("dataset: {source} ({} items, showing {})", corpus.tasks.len(), tasks.len());
    println!("budget:  {max_tokens} tok/item, {timeout_secs}s request timeout");
    if use_tools {
        println!(
            "tools:   python loop, up to {max_tool_steps} steps, {}s each",
            tool_timeout.as_secs()
        );
    }
    println!();

    let mut ledger = Ledger::in_memory(Box::new(FixedClock("2026-09-13T00:00:00Z".into()))).unwrap();
    let cfg = EpisodeConfig::default();

    let mut solved = 0usize;
    let mut completed = 0usize; // items the model actually answered (and were graded)
    let mut failed = 0usize; // server/transport failures — not the model's answer
    let mut conf_sum = 0.0;
    let mut brier_sum = 0.0;
    // domain -> (correct, total)
    let mut by_domain: std::collections::BTreeMap<String, (usize, usize)> = Default::default();

    // Resumable runs: if RESUME names a file, load per-item results already in it
    // and skip those questions, appending each new one as it completes. A dropped
    // Colab runtime then costs only the unfinished items — reconnect, set the new
    // SAMARITAN_URL, rerun with the same RESUME, and it continues. Failed items are
    // NOT written, so they retry next run. (Delete the file for a fresh run, e.g.
    // after the model changes.) Keyed by the question text, so it survives a new
    // tunnel URL and a reordered file, but not an edited question.
    let resume_path = std::env::var("RESUME").ok().filter(|s| !s.is_empty());
    let mut done: std::collections::HashMap<String, (bool, f64, u64)> = Default::default();
    if let Some(p) = &resume_path {
        if let Ok(txt) = std::fs::read_to_string(p) {
            for line in txt.lines().filter(|l| !l.trim().is_empty()) {
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
                    if let Some(q) = v["question"].as_str() {
                        done.insert(
                            q.to_string(),
                            (
                                v["correct"].as_bool().unwrap_or(false),
                                v["confidence"].as_f64().unwrap_or(0.5),
                                v["tokens"].as_u64().unwrap_or(0),
                            ),
                        );
                    }
                }
            }
            if !done.is_empty() {
                println!("resume: {} item(s) already answered in {p}\n", done.len());
            }
        }
    }
    let mut resume_file = resume_path.as_ref().map(|p| {
        std::fs::OpenOptions::new().create(true).append(true).open(p).expect("open RESUME file")
    });

    for (i, t) in tasks.iter().enumerate() {
        // Answered on a prior run? Count the cached result and skip the model call.
        if let Some(&(correct, conf, tokens)) = done.get(&t.prompt) {
            completed += 1;
            if correct {
                solved += 1;
            }
            conf_sum += conf;
            brier_sum += (conf - if correct { 1.0 } else { 0.0 }).powi(2);
            let e = by_domain.entry(t.domain().to_string()).or_insert((0, 0));
            e.1 += 1;
            if correct {
                e.0 += 1;
            }
            println!(
                "  q{:>2} [{:<9}] {}  conf {:.2}  ({} tok)  [cached]",
                i + 1,
                t.domain(),
                if correct { "OK  " } else { "MISS" },
                conf,
                tokens,
            );
            continue;
        }

        let out = match run_reasoning_episode(t, solver, &mut ledger, &cfg) {
            Ok(o) => o,
            Err(e) => {
                println!("  q{:>2} [{}]  ERROR: {e}", i + 1, t.domain());
                continue;
            }
        };

        // A server/transport failure — a dropped tunnel, a 5xx — is not a wrong
        // answer. Surface it, count it separately, and keep it out of accuracy and
        // calibration: otherwise a flaky endpoint reads as the model missing, and
        // one tunnel death late in a run tanks the whole score.
        if let Ending::AgentFailed { detail } = &out.ending {
            failed += 1;
            let d: String = detail.replace('\n', " ").chars().take(300).collect();
            println!("  q{:>2} [{:<9}] FAIL  {}", i + 1, t.domain(), d);
            continue;
        }

        completed += 1;
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
        // On a real miss, show what the model actually answered — the fastest way
        // to tell a reasoning error from an extraction/grading edge.
        if !correct {
            if let Some(a) = &out.answer {
                let a: String = a.replace('\n', " ").chars().take(100).collect();
                println!("        got: {a:?}");
            }
        }

        // Persist this answer so a later crash resumes past it. Flushed per item,
        // so progress survives a hard runtime drop mid-run.
        if let Some(f) = resume_file.as_mut() {
            let rec = serde_json::json!({
                "question": t.prompt,
                "domain": t.domain(),
                "correct": correct,
                "confidence": conf,
                "tokens": out.tokens,
            });
            let _ = writeln!(f, "{rec}");
            let _ = f.flush();
        }
    }

    let n = completed.max(1);
    println!("\n── result ──────────────────────────────");
    println!(
        "accuracy:        {}/{} answered = {:.1}%",
        solved,
        completed,
        100.0 * solved as f64 / n as f64
    );
    if failed > 0 {
        println!("not answered:    {failed} lost to server/transport errors (excluded from accuracy)");
    }
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
