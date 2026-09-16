//! Emit a generated reasoning corpus — W7's data engine, first turn.
//!
//!     cargo run -p samaritan-curriculum --example generate
//!
//! Writes problems whose gold answers were computed by the generator (never
//! asserted by a model) as the reasoning JSONL every downstream consumer
//! already reads: `reason_eval` to measure, `selftrain_export` to distill.
//! Because the numbers, names, and structure are freshly sampled, the set is
//! post-cutoff by construction — the contamination-free counterpart to any
//! static corpus.
//!
//! Env:
//!   COUNT        problems to generate (default 300).
//!   DIFFICULTY   1..=5 (default 2). The ZPD knob.
//!   CALIBRATE    optional path to a progress JSONL (reason_eval's {correct}
//!                or selftrain_export's {solved} records); the observed
//!                solve-rate nudges DIFFICULTY one step toward the ~55-85%
//!                band before generating.
//!   FAMILIES     comma list of modpow,crt,recurrence,word,knights,automata,
//!                graph,divideconquer,sat,zebra (default: all ten, round-robin
//!                across math, logic and cs).
//!   SEED         RNG seed (default 1) — the whole set is re-derivable from it.
//!   SPLIT_LABEL  train (default) or held_out. A *held-out* generated slice is
//!                also the cleanest eval there is: same seed discipline, but
//!                use a DIFFERENT seed than any training slice.
//!   OUT          output path (default %USERPROFILE%\models\reasoning\
//!                generated-d<D>-s<SEED>.jsonl).

use samaritan_curriculum::{generate_set, zpd_adjust, Family};

fn main() {
    let count: usize = std::env::var("COUNT").ok().and_then(|s| s.parse().ok()).unwrap_or(300);
    let mut difficulty: u64 =
        std::env::var("DIFFICULTY").ok().and_then(|s| s.parse().ok()).unwrap_or(2);
    let seed: u64 = std::env::var("SEED").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    let split = std::env::var("SPLIT_LABEL").unwrap_or_else(|_| "train".into());

    let families: Vec<Family> = match std::env::var("FAMILIES") {
        Ok(list) => {
            let parsed: Vec<Family> = list.split(',').filter_map(Family::parse).collect();
            if parsed.is_empty() {
                eprintln!("FAMILIES parsed to nothing; expected a comma list like modpow,knights");
                std::process::exit(2);
            }
            parsed
        }
        Err(_) => Family::all(),
    };

    // ZPD: read an observed solve-rate and move the knob one step toward the
    // learning band. The curriculum tracks the solver.
    if let Ok(path) = std::env::var("CALIBRATE") {
        match std::fs::read_to_string(&path) {
            Ok(txt) => {
                let mut solved = 0usize;
                let mut total = 0usize;
                for line in txt.lines().filter(|l| !l.trim().is_empty()) {
                    if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
                        let flag = v["correct"].as_bool().or_else(|| v["solved"].as_bool());
                        if let Some(ok) = flag {
                            total += 1;
                            if ok {
                                solved += 1;
                            }
                        }
                    }
                }
                if total == 0 {
                    println!("calibrate: no graded records in {path}; keeping difficulty {difficulty}");
                } else {
                    let rate = solved as f64 / total as f64;
                    let adjusted = zpd_adjust(difficulty, rate);
                    println!(
                        "calibrate: {solved}/{total} solved = {:.0}% -> difficulty {} (was {})",
                        rate * 100.0,
                        adjusted,
                        difficulty
                    );
                    difficulty = adjusted;
                }
            }
            Err(e) => {
                eprintln!("could not read CALIBRATE file {path}: {e}");
                std::process::exit(1);
            }
        }
    }

    let out_path = std::env::var("OUT").unwrap_or_else(|_| {
        let home = std::env::var("USERPROFILE").or_else(|_| std::env::var("HOME")).unwrap_or_default();
        format!("{home}/models/reasoning/generated-d{difficulty}-s{seed}.jsonl")
    });

    let problems = generate_set(count, &families, difficulty, seed);

    let mut lines = Vec::with_capacity(problems.len());
    let mut by_domain: std::collections::BTreeMap<&str, usize> = Default::default();
    for p in &problems {
        *by_domain.entry(p.domain).or_insert(0) += 1;
        lines.push(
            serde_json::json!({
                "id": p.id,
                "question": p.question,
                "answer": p.answer,
                "answer_kind": p.answer_kind,
                "domain": p.domain,
                "split": split,
                // The generator's own solution path, when it records one. Null
                // rather than an empty list, so a consumer can tell "this family
                // is not instrumented yet" from "this problem needed no steps".
                "steps": if p.steps.is_empty() {
                    serde_json::Value::Null
                } else {
                    serde_json::Value::Array(
                        p.steps
                            .iter()
                            .map(|s| serde_json::json!({ "text": s.text, "value": s.value }))
                            .collect(),
                    )
                },
            })
            .to_string(),
        );
    }
    if let Some(dir) = std::path::Path::new(&out_path).parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    std::fs::write(&out_path, lines.join("\n") + "\n").expect("write generated jsonl");

    println!("generated {} problems (difficulty {difficulty}, seed {seed}, split {split})", problems.len());
    print!("families:");
    for f in &families {
        print!(" {}", f.name());
    }
    let traced = problems.iter().filter(|p| !p.steps.is_empty()).count();
    println!(
        "solution traces: {traced}/{} problems, {} step(s) total",
        problems.len(),
        problems.iter().map(|p| p.steps.len()).sum::<usize>()
    );
    println!("\nby domain:");
    for (d, n) in &by_domain {
        println!("  {d:<8} {n}");
    }
    println!("wrote {out_path}");
    println!("\nsample: {}", problems[0].question.chars().take(160).collect::<String>());
    println!("\nnext:");
    println!("  measure:  DATASET={out_path} cargo run -p samaritan-run --example reason_eval");
    println!("  distill:  DATASET={out_path} cargo run -p samaritan-run --example selftrain_export");
}
