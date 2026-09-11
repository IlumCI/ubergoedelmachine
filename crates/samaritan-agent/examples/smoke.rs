//! End-to-end smoke test: ask the local model for one decision.
//!
//! Proves the whole chain in one shot — prompt assembly, GBNF-constrained
//! decoding, JSON extraction, the decision language's validation, and
//! provenance stamping — against a real server rather than a fixture.
//!
//!     cargo run -p samaritan-agent --example smoke
//!
//! Expects `scripts\serve.ps1` to be running. Override with SAMARITAN_URL.

use std::time::Duration;

use samaritan_agent::{Agent, AgentConfig, Constrain, Prompt};
use samaritan_dsl::decision::PolicyVersion;
use samaritan_dsl::{Digest, hash_json};

fn main() {
    let base_url =
        std::env::var("SAMARITAN_URL").unwrap_or_else(|_| "http://127.0.0.1:8080/v1".into());

    let constrain = match std::env::var("SAMARITAN_CONSTRAIN").as_deref() {
        Ok("schema") => Constrain::Schema,
        Ok("none") => Constrain::None,
        _ => Constrain::Grammar,
    };

    let cfg = AgentConfig {
        base_url: base_url.clone(),
        model: "samaritan-playout".into(),
        temperature: 0.7,
        max_tokens: 700,
        constrain,
        timeout: Duration::from_secs(900),
        max_retries: 1,
        // Pinned, so this run is replayable and two runs are comparable.
        seed: Some(20260911),
        ..Default::default()
    };

    println!("server:    {base_url}");
    println!("constrain: {constrain:?}");

    // A real task shape: a failing assertion with the source withheld, which
    // is exactly what an episode looks like.
    let prompt = Prompt::new()
        .with_policy(&[
            "Read the failing test before editing anything; the assertion names the \
             expected behaviour."
                .to_string(),
            "Prefer the smallest change that makes the suite pass.".to_string(),
        ])
        .with_task(
            "fix addition sign error",
            "---- tests/add.rs ----\n\
             running 1 test\n\
             test t ... FAILED\n\n\
             assertion `left == right` failed\n  \
             left: 0\n right: 4\n  \
             at tests/add.rs:1",
        );

    println!(
        "prompt:    {} chars, {:.0}% cacheable prefix",
        prompt.system().len() + prompt.user().len(),
        prompt.cacheable_fraction() * 100.0
    );
    println!("authority: {:?}\n", prompt.authority());

    let agent = Agent::new(cfg);
    let policy_version = PolicyVersion(hash_json(&"smoke-v1"));
    let _ = Digest::ZERO;

    let started = std::time::Instant::now();
    match agent.decide(&prompt, policy_version) {
        Ok(c) => {
            let generated = c.usage.completion_tokens;
            let tps = if c.elapsed.as_secs_f64() > 0.0 {
                generated as f64 / c.elapsed.as_secs_f64()
            } else {
                0.0
            };
            println!("--- decision -------------------------------------------");
            println!("situation: {}", c.record.situation());
            println!("options:");
            for (i, o) in c.record.options().iter().enumerate() {
                let mark = if i == c.record.chosen_index() { "->" } else { "  " };
                println!("  {mark} {}", o.summary);
                println!("       {}", o.assessment);
            }
            println!("rationale: {}", c.record.rationale());
            println!(
                "predicts:  {}  (confidence {:.2})",
                c.record.prediction().outcome,
                c.record.prediction().confidence.get()
            );
            println!("actions:");
            for a in c.record.actions() {
                println!(
                    "  [{:?}/{:?}/{:?}] {}",
                    a.kind, a.reversibility, a.blast_radius, a.intent
                );
                println!("       {}", a.payload);
            }
            println!("authority: {:?}  (set by the harness, not the model)", c.record.authority());
            println!("--- cost -----------------------------------------------");
            println!(
                "prompt {} tok ({} cached), generated {} tok",
                c.usage.prompt_tokens, c.usage.cached_tokens, generated
            );
            println!(
                "{:.1}s wall, {:.2} tok/s, {} attempt(s), seed {}",
                c.elapsed.as_secs_f64(),
                tps,
                c.attempts,
                c.seed
            );
        }
        Err(e) => {
            eprintln!("failed after {:.1}s: {e}", started.elapsed().as_secs_f64());
            eprintln!("\nIs the server up?  .\\scripts\\serve.ps1 -Ngl 18");
            std::process::exit(1);
        }
    }
}
