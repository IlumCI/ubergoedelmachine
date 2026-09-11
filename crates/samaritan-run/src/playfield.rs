//! The seam between the runner and the expensive part.
//!
//! Everything the runner does that costs inference — playing episodes with a
//! policy, pairing a candidate against the incumbent — goes through this
//! trait. In production it wraps `samaritan_episode::run_episode` driving the
//! real agent; in a test it returns scripted outcomes. That separation is why
//! the entire orchestration — budget, arms, the certificate gate, the
//! degeneracy diagnosis — is testable on a machine that must not run the
//! model, which is the machine this was written on.

use samaritan_corpus::Split;
use samaritan_episode::EpisodeOutcome;
use samaritan_search::PolicyState;

/// Runs episodes on demand.
pub trait Playfield {
    /// Play `n` episodes on tasks from `split` under `policy`, returning what
    /// each produced. The implementation samples the tasks; the runner only
    /// says how many and from where.
    fn play(&mut self, policy: &PolicyState, split: Split, n: usize) -> Vec<EpisodeOutcome>;

    /// Play `n` *paired* episodes: the same tasks under both the incumbent and
    /// a candidate policy, returning `(candidate_solved, incumbent_solved)`
    /// per task.
    ///
    /// Pairing is not a convenience — it is what the certificate's validity
    /// rests on. The same task under both policies blocks out task difficulty,
    /// so a discordant pair is evidence about the *policies*, not about which
    /// tasks happened to be hard. An implementation that paired different
    /// tasks would silently destroy the guarantee, so this is one call rather
    /// than two the runner might forget to align.
    fn play_paired(
        &mut self,
        incumbent: &PolicyState,
        candidate: &PolicyState,
        split: Split,
        n: usize,
    ) -> Vec<(bool, bool)>;
}

/// The mean clean utility of a batch, and its calibration pairs.
///
/// A violated episode has no score, so it is excluded from the mean rather
/// than counted as zero — averaging a violation into a number is exactly the
/// lexicographic escape the kernel forbids, and it would be no better here.
pub fn summarise(outcomes: &[EpisodeOutcome]) -> BatchSummary {
    let mut score_sum = 0.0;
    let mut clean = 0u32;
    let mut violated = 0u32;
    let mut tokens = 0u64;
    let mut solved = 0u32;
    let mut pairs = Vec::new();

    for o in outcomes {
        tokens += o.tokens;
        if o.ending.solved() {
            solved += 1;
        }
        pairs.extend(o.predictions.iter().copied());
        match o.utility.score() {
            Some(s) => {
                score_sum += s;
                clean += 1;
            }
            None => violated += 1,
        }
    }

    BatchSummary {
        mean_utility: if clean == 0 { 0.0 } else { score_sum / clean as f64 },
        clean,
        violated,
        solved,
        tokens,
        calibration: pairs,
    }
}

/// What a batch of episodes came to.
#[derive(Debug, Clone, PartialEq)]
pub struct BatchSummary {
    /// Mean utility over the *clean* episodes only.
    pub mean_utility: f64,
    pub clean: u32,
    pub violated: u32,
    pub solved: u32,
    pub tokens: u64,
    /// `(confidence, resolved)` across every decision, for calibration.
    pub calibration: Vec<(f64, bool)>,
}

impl BatchSummary {
    pub fn episodes(&self) -> u32 {
        self.clean + self.violated
    }
}
