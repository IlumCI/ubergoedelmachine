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
//!   MAX_TOKENS        solver answer budget (default 32768 — a reasoning floor;
//!                     HLE questions run long and truncate before the final answer
//!                     at a small budget)
//!   HTTP_TIMEOUT_SECS per-request timeout override; the default scales with
//!                     MAX_TOKENS, because a big budget needs the wall-clock to
//!                     spend it or the request aborts as a transport timeout
//!   RESUME            path to a progress JSONL ({question, correct}); scored items
//!                     are skipped and new ones appended, so a dropped tunnel costs
//!                     only the unfinished items. Rerun with the same RESUME to
//!                     continue; delete it for a fresh run.
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
        std::env::var("MAX_TOKENS").ok().and_then(|s| s.parse().ok()).unwrap_or(32768);
    // Per-request timeout must track the token budget: at the A100's ~30-40 tok/s a
    // big answer needs many minutes, and a fixed 900s wall aborts it mid-generation
    // as `transport: timeout: global` (the q14 failure on AIME). Scale it;
    // HTTP_TIMEOUT_SECS overrides. Same formula as reason_eval.
    let timeout_secs: u64 = std::env::var("HTTP_TIMEOUT_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| 900.max(u64::from(max_tokens) / 20 + 300));
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
    println!("budget:  {max_tokens} tok/item, {timeout_secs}s request timeout");
    println!("scoring {} questions from {dataset}\n", questions.len());

    let solver = Agent::new(AgentConfig {
        base_url: base_url.clone(),
        api_key: api_key.clone(),
        model: model.clone(),
        temperature,
        max_tokens,
        repeat_penalty: 1.1,
        constrain: Constrain::None,
        timeout: Duration::from_secs(timeout_secs),
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

    // Resumable, like reason_eval: RESUME names a JSONL of {question, correct}
    // already scored; those are skipped and each new one is appended as it lands,
    // so a dropped tunnel mid-run costs only the unfinished items. Keyed by the
    // question text, so it survives a new tunnel URL. Delete the file for a fresh
    // run (e.g. after the model changes).
    let resume_path = std::env::var("RESUME").ok().filter(|s| !s.is_empty());
    let mut done: std::collections::HashMap<String, bool> = Default::default();
    if let Some(p) = &resume_path {
        if let Ok(txt) = std::fs::read_to_string(p) {
            for line in txt.lines().filter(|l| !l.trim().is_empty()) {
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
                    if let Some(qq) = v["question"].as_str() {
                        done.insert(qq.to_string(), v["correct"].as_bool().unwrap_or(false));
                    }
                }
            }
            if !done.is_empty() {
                println!("resume: {} item(s) already scored in {p}\n", done.len());
            }
        }
    }
    let mut resume_file = resume_path.as_ref().map(|p| {
        std::fs::OpenOptions::new().create(true).append(true).open(p).expect("open RESUME file")
    });

    let mut verdicts: Vec<bool> = Vec::with_capacity(questions.len());
    let mut failed = 0usize; // solver transport failures — not wrong answers
    for (i, q) in questions.iter().enumerate() {
        // Scored on a prior run? Count the cached verdict and skip the model call.
        if let Some(&correct) = done.get(&q.question) {
            verdicts.push(correct);
            println!("q{:>3}: {}  [cached]", i + 1, if correct { "OK  " } else { "MISS" });
            continue;
        }

        let prompt = format!(
            "{}\n\nReason it through, then end with your final answer, as concisely \
             as the question allows.",
            q.question
        );
        // A solver transport failure (a dropped tunnel, a 5xx) is not a wrong
        // answer: count it apart, keep it out of the score, and do NOT write it to
        // RESUME so it retries next run — otherwise a flaky endpoint reads as the
        // model missing and tanks the number. An empty-but-successful reply is a
        // real miss (the model answered nothing), so that still grades normally.
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
                failed += 1;
                let d: String = e.to_string().replace('\n', " ").chars().take(200).collect();
                println!("q{:>3}: FAIL  {d}", i + 1);
                continue;
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

        // Persist per item so a later crash resumes past it.
        if let Some(f) = resume_file.as_mut() {
            use std::io::Write as _;
            let rec = serde_json::json!({ "question": q.question.as_str(), "correct": correct });
            let _ = writeln!(f, "{rec}");
            let _ = f.flush();
        }
    }

    let result = tally_verdicts(&verdicts);
    println!(
        "\nscore ({method} grading): {}/{} = {:.1}%",
        result.correct,
        result.total,
        result.score() * 100.0
    );
    if failed > 0 {
        println!("not answered:  {failed} lost to server/transport errors (excluded from score)");
    }
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
