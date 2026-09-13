//! Score the model on an HLE-format JSONL and record the result.
//!
//!     HLE_DATASET=hle.jsonl cargo run -p samaritan-eval --example hle
//!
//! Expects an OpenAI-compatible server: the local one (scripts\serve.ps1) or a
//! remote one via SAMARITAN_URL — e.g. the A100 in docs/colab-remote.md. The
//! dataset is operator-supplied: HLE is gated and licensed, so this crate ships
//! the grader, not the questions (scripts\fetch-hle.ps1 -HfToken … fetches it).
//! Set LEDGER_DB to record an HleEvaluated row the milestone gate can read.
//!
//! Grading is a normalised string match by default — the deterministic floor,
//! which under-credits a correct answer phrased unusually. Set JUDGE=1 to grade
//! with an LLM judge instead, which is what real HLE does. The judge is shown the
//! gold answer and only decides equivalence; it never solves the question, and
//! the solver never sees the gold — so the harness still owns the oracle. If the
//! judge errors or won't commit on an item, that item falls back to the match, so
//! a run always yields a score.
//!
//! Env:
//!   HLE_DATASET       path to the JSONL (required)
//!   LIMIT             cap questions (default 50; the model is slow)
//!   SAMARITAN_URL     model server base (default http://127.0.0.1:8080/v1)
//!   SAMARITAN_API_KEY bearer key if the server wants one (e.g. a tunnelled A100)
//!   SAMARITAN_MODEL   served model name (default samaritan-playout; Ollama needs
//!                     the tag, e.g. samaritan-playout:latest)
//!   MAX_TOKENS        solver answer budget (default 8192 — a thinking model needs
//!                     room or it truncates before its final answer)
//!   TEMPERATURE       solver temperature (default 0.3)
//!   JUDGE             truthy to grade with the LLM judge
//!   JUDGE_URL / JUDGE_MODEL  judge endpoint (default: the solver's)
//!   LEDGER_DB         if set, append the HleEvaluated event here

use std::time::Duration;

use samaritan_agent::{Agent, AgentConfig, Constrain};
use samaritan_eval::{grade, judge_prompt, load, parse_verdict, tally_verdicts, GradeMethod};
use samaritan_ledger::{Actor, Event, FixedClock, Ledger};

/// Everything after the last `</think>` — the model's answer proper. A thinking
/// model wraps its reasoning in `<think>…</think>`; grading and the judge want the
/// final answer, not the trace.
fn answer_body(reply: &str) -> &str {
    match reply.rfind("</think>") {
        Some(i) => reply[i + "</think>".len()..].trim(),
        None => reply.trim(),
    }
}

fn truthy(v: &str) -> bool {
    matches!(v.trim().to_lowercase().as_str(), "1" | "true" | "yes" | "on")
}

fn main() {
    let dataset = std::env::var("HLE_DATASET").unwrap_or_else(|_| {
        eprintln!(
            "set HLE_DATASET to a JSONL of HLE-format rows \
             ({{question, answer, answer_type}}). HLE is gated and licensed; \
             supply your own copy (scripts\\fetch-hle.ps1 -HfToken …)."
        );
        std::process::exit(2);
    });
    let limit: usize = std::env::var("LIMIT").ok().and_then(|s| s.parse().ok()).unwrap_or(50);
    let base_url =
        std::env::var("SAMARITAN_URL").unwrap_or_else(|_| "http://127.0.0.1:8080/v1".into());
    let api_key = std::env::var("SAMARITAN_API_KEY").unwrap_or_default();
    // Ollama tags its models and matches the id exactly, so a remote Ollama solver
    // wants SAMARITAN_MODEL=samaritan-playout:latest. Bare name for local llama.cpp.
    let model = std::env::var("SAMARITAN_MODEL").unwrap_or_else(|_| "samaritan-playout".into());
    let max_tokens: u32 =
        std::env::var("MAX_TOKENS").ok().and_then(|s| s.parse().ok()).unwrap_or(8192);
    let temperature: f64 =
        std::env::var("TEMPERATURE").ok().and_then(|s| s.parse().ok()).unwrap_or(0.3);
    let use_judge = std::env::var("JUDGE").ok().map(|v| truthy(&v)).unwrap_or(false);

    let text = std::fs::read_to_string(&dataset).unwrap_or_else(|e| {
        eprintln!("could not read {dataset}: {e}");
        std::process::exit(1);
    });
    let mut questions = load(&text).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1);
    });
    questions.truncate(limit);

    let method = if use_judge { GradeMethod::LlmJudge } else { GradeMethod::NormalizedMatch };
    println!("server:  {base_url}");
    println!("model:   {model}");
    println!("grading: {method}");
    println!("scoring {} questions from {dataset}\n", questions.len());

    let solver = Agent::new(AgentConfig {
        base_url: base_url.clone(),
        api_key: api_key.clone(),
        model: model.clone(),
        temperature,
        max_tokens,
        repeat_penalty: 1.1,
        constrain: Constrain::None,
        timeout: Duration::from_secs(900),
        max_retries: 1,
        seed: Some(1),
        ..Default::default()
    });

    // The judge, when enabled: same endpoint/model by default (a strong model
    // deciding equivalence), overridable. Low temperature and a modest budget —
    // the verdict is short; if it truncates before deciding, we fall back to the
    // deterministic match.
    let judge = if use_judge {
        let jurl = std::env::var("JUDGE_URL").unwrap_or_else(|_| base_url.clone());
        let jmodel = std::env::var("JUDGE_MODEL").unwrap_or_else(|_| model.clone());
        println!("judge:   {jmodel} @ {jurl}\n");
        Some(Agent::new(AgentConfig {
            base_url: jurl,
            api_key: api_key.clone(),
            model: jmodel,
            temperature: 0.0,
            max_tokens: 2048,
            repeat_penalty: 1.0,
            constrain: Constrain::None,
            timeout: Duration::from_secs(900),
            max_retries: 1,
            seed: Some(1),
            ..Default::default()
        }))
    } else {
        None
    };

    let mut verdicts: Vec<bool> = Vec::with_capacity(questions.len());
    for (i, q) in questions.iter().enumerate() {
        let prompt = format!(
            "{}\n\nReason it through, then end with your final answer, as concisely \
             as the question allows.",
            q.question
        );
        let reply = match solver.complete(
            "You are taking a hard exam. Answer each question precisely.",
            &prompt,
            None,
            None,
            temperature,
            1,
        ) {
            Ok((content, _)) => content,
            Err(e) => {
                eprintln!("q{}: model error: {e}", i + 1);
                String::new()
            }
        };
        let ans = answer_body(&reply).to_string();

        // Grade: the judge if enabled, else the deterministic match. On a judge
        // error or an undecided verdict, fall back to the match so a run always
        // produces a score.
        let correct = match &judge {
            Some(j) => {
                let (js, ju) = judge_prompt(q, &ans);
                match j.complete(&js, &ju, None, None, 0.0, 1) {
                    Ok((v, _)) => parse_verdict(&v).unwrap_or_else(|| grade(q, &ans)),
                    Err(e) => {
                        eprintln!("q{}: judge error: {e} (falling back to match)", i + 1);
                        grade(q, &ans)
                    }
                }
            }
            None => grade(q, &ans),
        };
        verdicts.push(correct);

        let shown: String = ans.lines().next_back().unwrap_or("").trim().chars().take(80).collect();
        println!("q{:>3}: {}  {:?}", i + 1, if correct { "OK  " } else { "MISS" }, shown);
    }

    let result = tally_verdicts(&verdicts);
    println!(
        "\nscore ({method} grading): {}/{} = {:.1}%",
        result.correct,
        result.total,
        result.score() * 100.0
    );
    if use_judge {
        println!("(LLM-judged — closer to official HLE than a string match, but the judge \
                  is itself a model; not an official score.)");
    }

    if let Ok(path) = std::env::var("LEDGER_DB") {
        let mut ledger =
            Ledger::open_with_clock(&path, Box::new(FixedClock("2026-09-12T00:00:00Z".into())))
                .expect("open ledger");
        ledger
            .append(
                Actor::System,
                &Event::HleEvaluated {
                    score: result.score(),
                    questions: result.total,
                    model: model.clone(),
                },
            )
            .expect("append");
        println!("recorded HleEvaluated to {path}");
    }
}
