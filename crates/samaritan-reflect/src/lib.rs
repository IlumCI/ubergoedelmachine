//! Turning a run's outcomes into candidate self-modifications.
//!
//! This is the level-1 mutation *source* — the answer to "where does a
//! lesson come from". The nested search ([`samaritan_search`]) knows how to
//! search over mutations; the certificate ([`samaritan_cert`]) knows how to
//! decide whether one may be kept. Neither knows how to *propose* one from a
//! run's history. That is this crate.
//!
//! # Mined, not authored
//!
//! Every candidate here is derived from ledger statistics, and its text is a
//! rendering of the statistic rather than a sentence the model wrote. This is
//! a deliberate design commitment, and it is worth being explicit about why,
//! because the easier design — ask the model "what did you learn?" — is
//! tempting and wrong:
//!
//! - **The map is not the territory.** When a model states "I learned X",
//!   that sentence is a lossy, often confabulated summary of whatever actually
//!   helped. Language models report their own causal reasoning poorly, so the
//!   stated lesson and the useful lesson routinely diverge.
//! - **Self-report is an attack surface.** A model that knows plausible-
//!   sounding lessons get kept can farm them — reward-hacking the reflection
//!   pass with text that reads well and changes nothing. Mining from outcomes
//!   starves that: a lesson has to correspond to a real pattern in what
//!   happened, and it still has to clear the certificate on held-out tasks
//!   before it is committed.
//! - **It matches how the rest of the system already thinks.** The
//!   certificate distrusts stated reasons and measures outcomes. Lesson
//!   memory should reason the same way, or the two halves of the harness are
//!   working from different epistemologies.
//!
//! So a [`Candidate`] carries its **evidence** — how many observations
//! support it and how strong the effect is — alongside the mutation. That
//! evidence is what the search ranks by and what a human reads; the mutation
//! is what gets tested. The model's own narration is treated as untrusted
//! decoration, never as the source of truth.

pub mod domain;
pub mod grammar;
pub mod observation;

use samaritan_dsl::{Knob, Mutation};
use serde::{Deserialize, Serialize};

pub use domain::{SelfModDomain, search_mined};
pub use grammar::{GrammarDomain, grammar_ops, search_grammar};
pub use observation::{Corpus, Observation};

/// A proposed self-modification, with the evidence that produced it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Candidate {
    pub mutation: Mutation,
    /// One line, generated from the statistic, for a human reading the
    /// ledger. Never model-authored.
    pub rationale: String,
    /// How many observations back this candidate. A lesson mined from three
    /// episodes is a guess; one mined from three hundred is a pattern, and
    /// the search should be able to tell them apart.
    pub support: u64,
    /// Effect size in `[0, 1]`: how strongly the pattern held. A recurring
    /// breach has effect 1.0; a mild calibration skew, less.
    pub effect: f64,
}

impl Candidate {
    /// A crude priority for ordering candidates before the search refines
    /// them: evidence times effect. Deliberately not the final say — the
    /// certificate is — but a sensible order to try things in.
    pub fn weight(&self) -> f64 {
        (self.support as f64).sqrt() * self.effect
    }
}

/// How aggressively to mine.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MineConfig {
    /// Minimum observations before a calibration pattern is proposed. Below
    /// this the "pattern" is just noise, and a lesson mined from noise is
    /// worse than none — it spends prompt budget and misfires.
    pub min_support: u64,
    /// Miscalibration rate above which a hedging lesson is proposed.
    pub overconfidence_threshold: f64,
    /// Breaches of one kind before a lesson is proposed about it. One breach
    /// is an experiment; several is a habit worth naming.
    pub breach_repeat_threshold: u64,
}

impl Default for MineConfig {
    fn default() -> Self {
        Self {
            min_support: 20,
            overconfidence_threshold: 0.3,
            breach_repeat_threshold: 2,
        }
    }
}

/// Mine a run's outcomes into candidate mutations.
///
/// Pure over the distilled [`Corpus`], so the whole mechanism is testable
/// against a synthetic ledger with no model and no live run. That is the
/// point: the reflection pass must be checkable in isolation, or a bug in it
/// would only ever surface as a slightly worse self-improvement curve, which
/// is exactly the kind of failure nobody catches.
pub fn mine(corpus: &Corpus, cfg: &MineConfig) -> Vec<Candidate> {
    let mut out = Vec::new();
    out.extend(mine_calibration(corpus, cfg));
    out.extend(mine_breaches(corpus, cfg));
    // Highest-evidence first, so a caller that takes the top few gets the
    // best-supported ones.
    out.sort_by(|a, b| b.weight().partial_cmp(&a.weight()).unwrap_or(std::cmp::Ordering::Equal));
    out
}

/// Calibration mining: if the agent is confident-and-wrong too often, propose
/// tightening the calibration ceiling *and* a lesson about hedging.
///
/// This is the mining the discussion singled out as high-value: calibration
/// is metacognition that compresses to a scalar without loss, so a lesson
/// about it is honest in a way a content lesson is not.
fn mine_calibration(corpus: &Corpus, cfg: &MineConfig) -> Vec<Candidate> {
    let confident: Vec<&Observation> = corpus
        .observations
        .iter()
        .filter(|o| o.stated_confidence >= 0.8)
        .collect();

    if (confident.len() as u64) < cfg.min_support {
        return Vec::new();
    }

    let misses = confident.iter().filter(|o| !o.resolved).count();
    let miss_rate = misses as f64 / confident.len() as f64;

    if miss_rate < cfg.overconfidence_threshold {
        return Vec::new();
    }

    // The lesson text states the measured rate. It is a rendering of the
    // number, not a claim the model made about itself.
    let pct = (miss_rate * 100.0).round() as u64;
    vec![
        Candidate {
            mutation: Mutation::LessonAdd {
                text: format!(
                    "You have been wrong on {pct}% of the predictions you rated 0.8 or higher. \
                     State a lower confidence unless the evidence is decisive."
                ),
            },
            rationale: format!(
                "mined: {misses}/{} high-confidence predictions missed ({pct}%)",
                confident.len()
            ),
            support: confident.len() as u64,
            effect: miss_rate,
        },
        Candidate {
            mutation: Mutation::ThresholdSet {
                knob: Knob::CalibrationCeiling,
                // Tighten toward the frozen floor. The kernel clamps this, so
                // an over-aggressive value simply lands at the bound.
                value: 0.1,
            },
            rationale: "mined: overconfidence should not be able to earn autonomy".into(),
            support: confident.len() as u64,
            effect: miss_rate,
        },
    ]
}

/// Breach mining: a refusal reason the agent has hit more than once becomes a
/// lesson naming it.
///
/// A breach is the sharpest possible signal — the agent reached for something
/// the frozen core refused — so a recurring one is worth a full-effect lesson.
fn mine_breaches(corpus: &Corpus, cfg: &MineConfig) -> Vec<Candidate> {
    let mut out = Vec::new();
    // Stable order over the map so the mined set is reproducible.
    let mut breaches: Vec<(&String, &u64)> = corpus.breach_counts.iter().collect();
    breaches.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));

    for (reason, count) in breaches {
        if *count < cfg.breach_repeat_threshold {
            continue;
        }
        out.push(Candidate {
            mutation: Mutation::LessonAdd {
                text: format!(
                    "An action was refused {count} times for the same reason: {reason}. \
                     Do not propose it again."
                ),
            },
            rationale: format!("mined: {count} repeated breaches of one kind"),
            support: *count,
            // A breach is categorical: it happened, and repeatedly. Full
            // effect.
            effect: 1.0,
        });
    }
    out
}
