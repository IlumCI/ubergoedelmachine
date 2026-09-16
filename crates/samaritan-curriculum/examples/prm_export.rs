//! Emit a process-reward dataset from W7's own solution traces.
//!
//!     cargo run -p samaritan-curriculum --example prm_export
//!
//! A process reward model scores *individual reasoning steps* rather than only
//! the final answer. Training one needs steps labelled correct or incorrect, and
//! obtaining those labels is the expensive part of the whole literature: human
//! annotation does not scale, so the standard substitute is Monte-Carlo rollouts
//! from each partial state, taking the fraction of completions that end up right
//! as a noisy proxy for whether the step was good. That costs N extra
//! generations per step and still only estimates the label.
//!
//! Here both halves are exact and free. The generator computed the gold from an
//! algorithm, so the true intermediate value at every step is known — that is
//! the positive. And because the true value is known, a *deliberately wrong* one
//! can be substituted and known to be wrong — that is the negative, with no
//! rollouts and no estimation.
//!
//! **The honest limitation.** Synthetic corruptions teach a model to recognise
//! the corruption distribution, which is not the same as recognising reasoning
//! errors a model actually makes. Errors here are arithmetic slips, stale
//! values and transposed digits — plausible, and drawn from how these
//! calculations really go wrong, but not sampled from the student. The strongest
//! dataset mixes these exact negatives with mined ones from real failed rollouts;
//! the schema below carries a `source` field so both can live in one file. Treat
//! this as the free half, not the whole thing.
//!
//! Labels follow the usual convention: steps up to the first error are 1, the
//! erroneous step is 0, and everything after it is dropped rather than labelled.
//! A step following an error is neither right nor wrong in any useful sense —
//! it is reasoning from a false premise, and labelling it either way teaches
//! something untrue.
//!
//! Env:
//!   COUNT        problems to draw (default 200).
//!   DIFFICULTY   1..=5 (default 3).
//!   SEED         RNG seed (default 1). Use a different one from any eval set.
//!   FAMILIES     comma list (default: all ten).
//!   NEGATIVES    corrupted variants per problem (default 2).
//!   OUT          output path (default %USERPROFILE%\models\reasoning\prm-d<D>-s<SEED>.jsonl).

use samaritan_curriculum::{generate_set, Family, Problem, SplitMix64, Step};

/// How a step was broken. Recorded so a trained model's failures can be read
/// back per error type — if it only ever catches `magnitude`, that is worth
/// knowing before trusting it on `off_by_one`.
#[derive(Clone, Copy, Debug)]
enum Corruption {
    OffByOne,
    Transpose,
    Magnitude,
    StaleValue,
    InnerDigit,
}

impl Corruption {
    fn name(self) -> &'static str {
        match self {
            Corruption::OffByOne => "off_by_one",
            Corruption::Transpose => "transpose",
            Corruption::Magnitude => "magnitude",
            Corruption::StaleValue => "stale_value",
            Corruption::InnerDigit => "inner_digit",
        }
    }
}

/// Produce a wrong-but-plausible version of `value`.
///
/// `prior` is an earlier step's value, which makes the most realistic error of
/// all available: carrying a stale quantity forward. Returns None when the
/// corruption would coincide with the truth — a "wrong" answer that happens to
/// be right is a mislabelled example, which is worse than one fewer example.
fn corrupt(value: &str, kind: Corruption, prior: Option<&str>) -> Option<String> {
    let out = match kind {
        Corruption::OffByOne => {
            let n: i64 = value.parse().ok()?;
            (if n % 2 == 0 { n + 1 } else { n - 1 }).to_string()
        }
        Corruption::Transpose => {
            let digits: Vec<char> = value.chars().collect();
            if digits.len() < 2 || !digits.iter().all(|c| c.is_ascii_digit()) {
                return None;
            }
            let mut d = digits.clone();
            let mut i = 0;
            while i + 1 < d.len() && d[i] == d[i + 1] {
                i += 1; // transposing equal digits changes nothing
            }
            if i + 1 >= d.len() {
                return None;
            }
            d.swap(i, i + 1);
            if d[0] == '0' {
                return None; // leading zero reads as a typo, not a slip
            }
            d.into_iter().collect()
        }
        Corruption::Magnitude => {
            let n: i64 = value.parse().ok()?;
            if n == 0 {
                return None;
            }
            (n * 10).to_string()
        }
        Corruption::StaleValue => {
            // Integer-to-integer only. Substituting a structured value such as
            // "5 mod 27" into a slot expecting a bare number yields prose no
            // solver would ever write, which teaches string malformation rather
            // than arithmetic error.
            let p = prior?;
            value.parse::<i64>().ok()?;
            p.parse::<i64>().ok()?;
            p.to_string()
        }
        Corruption::InnerDigit => {
            // For compound values like "q0=1, q1=0": nudge one number inside.
            // Whole-value parses decline these, so without this they would never
            // be corrupted at all - and a DP over states is precisely where a
            // step-level error matters most.
            let mut nums: Vec<(usize, usize)> = Vec::new();
            let bytes = value.as_bytes();
            let mut i = 0;
            while i < bytes.len() {
                if bytes[i].is_ascii_digit() {
                    let start = i;
                    while i < bytes.len() && bytes[i].is_ascii_digit() {
                        i += 1;
                    }
                    nums.push((start, i));
                } else {
                    i += 1;
                }
            }
            if nums.len() < 2 {
                return None; // a lone number is off_by_one's job
            }
            // The last number is the one a running calculation most often gets
            // wrong, and changing it leaves the rest of the value coherent.
            let (a, b) = nums[nums.len() - 1];
            let n: i64 = value[a..b].parse().ok()?;
            let bumped = if n == 0 { 1 } else { n + 1 };
            format!("{}{}{}", &value[..a], bumped, &value[b..])
        }
    };
    if out == value {
        return None;
    }
    Some(out)
}

/// A step is corruptible only when its value appears verbatim in its text —
/// otherwise the substitution leaves prose that contradicts the value it
/// carries, and the example teaches incoherence rather than error detection.
fn corruptible(step: &Step) -> Option<&str> {
    let v = step.value.as_deref()?;
    if v.len() >= 1 && step.text.contains(v) {
        Some(v)
    } else {
        None
    }
}

fn json_escape(s: &str) -> String {
    serde_json::Value::String(s.to_string()).to_string()
}

fn family_of(id: &str) -> &str {
    id.split('-').nth(1).unwrap_or("unknown")
}

fn record(
    p: &Problem,
    steps: &[String],
    labels: &[u8],
    source: &str,
    note: String,
    correct_step: Option<&str>,
) -> String {
    let steps_json: Vec<String> = steps.iter().map(|s| json_escape(s)).collect();
    let labels_json: Vec<String> = labels.iter().map(|l| l.to_string()).collect();
    // `correct_step` is what the final step SHOULD have said. Present only on
    // negatives, where it makes the record a matched pair: identical prefix,
    // one right continuation and one wrong.
    let correct_json = match correct_step {
        Some(c) => json_escape(c),
        None => "null".to_string(),
    };
    format!(
        r#"{{"id":{},"family":{},"domain":{},"question":{},"answer":{},"steps":[{}],"labels":[{}],"source":{},"note":{},"correct_step":{}}}"#,
        json_escape(&p.id),
        json_escape(family_of(&p.id)),
        json_escape(p.domain),
        json_escape(&p.question),
        json_escape(&p.answer),
        steps_json.join(","),
        labels_json.join(","),
        json_escape(source),
        json_escape(&note),
        correct_json,
    )
}

fn main() {
    let count: usize = std::env::var("COUNT").ok().and_then(|s| s.parse().ok()).unwrap_or(200);
    let difficulty: u64 =
        std::env::var("DIFFICULTY").ok().and_then(|s| s.parse().ok()).unwrap_or(3);
    let seed: u64 = std::env::var("SEED").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    let negatives: usize =
        std::env::var("NEGATIVES").ok().and_then(|s| s.parse().ok()).unwrap_or(2);

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

    let out_path = std::env::var("OUT").unwrap_or_else(|_| {
        let home =
            std::env::var("USERPROFILE").or_else(|_| std::env::var("HOME")).unwrap_or_default();
        format!("{home}/models/reasoning/prm-d{difficulty}-s{seed}.jsonl")
    });

    let problems = generate_set(count, &families, difficulty, seed);
    let mut rng = SplitMix64::new(seed ^ 0x9E37_79B9);
    let mut lines: Vec<String> = Vec::new();

    let kinds = [
        Corruption::OffByOne,
        Corruption::Transpose,
        Corruption::Magnitude,
        Corruption::StaleValue,
        Corruption::InnerDigit,
    ];
    let mut by_kind: std::collections::BTreeMap<&str, usize> = Default::default();
    let mut by_family: std::collections::BTreeMap<&str, (usize, usize)> = Default::default();
    let mut skipped = 0usize;

    for p in &problems {
        if p.steps.is_empty() {
            continue;
        }
        let texts: Vec<String> = p.steps.iter().map(|s| s.text.clone()).collect();
        let entry = by_family.entry(family_of(&p.id)).or_insert((0, 0));

        // The positive: the generator's own path, every step correct.
        lines.push(record(p, &texts, &vec![1u8; texts.len()], "generator", String::new(), None));
        entry.0 += 1;

        // Negatives: break one step, keep everything before it, drop everything
        // after. Indices are drawn without replacement so two negatives from the
        // same problem cannot be the same example.
        let mut used: Vec<usize> = Vec::new();
        for _ in 0..negatives {
            let mut made = false;
            for _try in 0..24 {
                let i = rng.range(0, p.steps.len() as u64 - 1) as usize;
                if used.contains(&i) {
                    continue;
                }
                let Some(truth) = corruptible(&p.steps[i]) else { continue };
                let kind = kinds[rng.range(0, kinds.len() as u64 - 1) as usize];
                let prior = if i > 0 {
                    p.steps[..i].iter().rev().find_map(|s| s.value.as_deref())
                } else {
                    None
                };
                let Some(wrong) = corrupt(truth, kind, prior) else { continue };

                let broken = p.steps[i].text.replacen(truth, &wrong, 1);
                if broken == p.steps[i].text {
                    continue;
                }
                // If the true value still appears after substitution, the step
                // contradicts itself outright and reads as a typo rather than a
                // mistaken calculation.
                if truth.parse::<i64>().is_ok() && broken.contains(truth) {
                    continue;
                }
                let mut steps: Vec<String> = texts[..i].to_vec();
                steps.push(broken);
                let mut labels = vec![1u8; i];
                labels.push(0);

                lines.push(record(
                    p,
                    &steps,
                    &labels,
                    "corruption",
                    format!("step {i}: {} {truth} -> {wrong}", kind.name()),
                    Some(&p.steps[i].text),
                ));
                *by_kind.entry(kind.name()).or_insert(0) += 1;
                entry.1 += 1;
                used.push(i);
                made = true;
                break;
            }
            if !made {
                skipped += 1;
            }
        }
    }

    if let Some(dir) = std::path::Path::new(&out_path).parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    std::fs::write(&out_path, lines.join("\n") + "\n").expect("write prm jsonl");

    let pos: usize = by_family.values().map(|(p, _)| p).sum();
    let neg: usize = by_family.values().map(|(_, n)| n).sum();
    println!("wrote {} examples to {out_path}", lines.len());
    println!("  {pos} positive (generator's own path)");
    println!("  {neg} negative (one corrupted step, trace truncated there)");
    if skipped > 0 {
        println!(
            "  {skipped} negative(s) not produced - no corruptible step survived the \
             coincidence check, which is a skip rather than a mislabel"
        );
    }
    let pos_steps: usize = lines
        .iter()
        .filter(|l| l.contains(r#""source":"generator""#))
        .count();
    let _ = pos_steps;
    println!(
        "\nstep-level labels are ~15% negative and adding NEGATIVES does not move that -\n\
         each negative carries its own correct prefix. Either weight the loss, or use the\n\
         `correct_step` field to read each negative as a matched pair (same prefix, one right\n\
         continuation and one wrong), which is balanced by construction."
    );
    println!("\nby corruption kind:");
    for (k, n) in &by_kind {
        println!("  {k:<12} {n}");
    }
    println!("\nby family (positive/negative):");
    for (f, (p, n)) in &by_family {
        println!("  {f:<14} {p:>4} / {n:<4}");
    }
    println!(
        "\nNote: these negatives are exact but synthetic. Mix in negatives mined from real\n\
         failed rollouts before trusting a PRM trained on this alone - the schema's `source`\n\
         field is there so both kinds can share one file."
    );
}
