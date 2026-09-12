//! Score the local model on an HLE-format JSONL and record the result.
//!
//!     HLE_DATASET=hle.jsonl cargo run -p samaritan-eval --example hle
//!
//! Expects a local model server (scripts\serve.ps1). The dataset is
//! operator-supplied: HLE is gated and licensed, so this crate ships the
//! grader, not the questions. Set LEDGER_DB to record an HleEvaluated row the
//! milestone gate can read; otherwise the score is just printed.
//!
//! Env:
//!   HLE_DATASET  path to the JSONL (required)
//!   LIMIT        cap the number of questions (default 50; the model is slow)
//!   LEDGER_DB    if set, append the HleEvaluated event here
//!   SAMARITAN_URL / model server base (default http://127.0.0.1:8080/v1)

use std::time::Duration;

use samaritan_agent::{Agent, AgentConfig, Constrain};
use samaritan_eval::{load, tally};
use samaritan_ledger::{Actor, Event, FixedClock, Ledger};

fn main() {
    let dataset = std::env::var("HLE_DATASET").unwrap_or_else(|_| {
        eprintln!(
            "set HLE_DATASET to a JSONL of HLE-format rows \
             ({{question, answer, answer_type}}). HLE is gated and licensed; \
             supply your own copy."
        );
        std::process::exit(2);
    });
    let limit: usize = std::env::var("LIMIT").ok().and_then(|s| s.parse().ok()).unwrap_or(50);
    let base_url =
        std::env::var("SAMARITAN_URL").unwrap_or_else(|_| "http://127.0.0.1:8080/v1".into());
    let model = "samaritan-playout".to_string();

    let text = std::fs::read_to_string(&dataset).unwrap_or_else(|e| {
        eprintln!("could not read {dataset}: {e}");
        std::process::exit(1);
    });
    let mut questions = load(&text).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1);
    });
    questions.truncate(limit);
    println!("scoring {} questions from {dataset}\n", questions.len());

    let agent = Agent::new(AgentConfig {
        base_url,
        model: model.clone(),
        // Low temperature: an evaluation wants the model's best single answer,
        // not a creative one. Free-form — HLE answers are not a fixed grammar.
        temperature: 0.2,
        max_tokens: 512,
        repeat_penalty: 1.1,
        constrain: Constrain::None,
        timeout: Duration::from_secs(900),
        max_retries: 1,
        seed: Some(1),
        ..Default::default()
    });

    let mut answers: Vec<String> = Vec::with_capacity(questions.len());
    for (i, q) in questions.iter().enumerate() {
        let prompt = format!(
            "{}\n\nAnswer as concisely as possible. Give only the final answer, \
             nothing else.",
            q.question
        );
        let reply = match agent.complete(
            "You are taking a hard exam. Answer each question precisely.",
            &prompt,
            None,
            None,
            0.2,
            1,
        ) {
            Ok((content, _)) => content,
            Err(e) => {
                eprintln!("q{}: model error: {e}", i + 1);
                String::new()
            }
        };
        println!("q{:>3}: {}", i + 1, reply.lines().next().unwrap_or("").trim());
        answers.push(reply);
    }

    let result = tally(&questions, &answers);
    println!(
        "\nscore: {}/{} = {:.1}%",
        result.correct,
        result.total,
        result.score() * 100.0
    );

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
                    model,
                },
            )
            .expect("append");
        println!("recorded HleEvaluated to {path}");
    }
}
