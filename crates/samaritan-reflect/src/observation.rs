//! Reading the ledger back into a shape the miner can count.
//!
//! The ledger is an event stream; learning needs a per-decision view — what
//! was predicted, how sure, what class of action was taken, and whether it
//! worked. This module joins the scattered events into that view and nothing
//! more. It draws no conclusions; the miner does that.
//!
//! One discipline runs through the whole crate and starts here: **we read
//! outcomes, never the agent's account of them.** An `Observation` is built
//! from the `OutcomeObserved` row the oracle wrote and the `EpisodeScored`
//! row the harness wrote, joined to the prediction the agent stated *before*
//! it acted. The agent's later narration of what it learned is not consulted.
//! A system that mined the model's self-report would be learning what the
//! model finds easy to say, which is not the same as what is true.

use std::collections::HashMap;

use samaritan_ledger::{Event, Ledger, LedgerError};
use serde::{Deserialize, Serialize};

/// One decision, joined to how it turned out.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Observation {
    pub decision: String,
    /// Confidence the agent stated *before* acting.
    pub stated_confidence: f64,
    /// Whether the predicted outcome actually occurred, per the oracle.
    pub resolved: bool,
    /// Whether the episode was scored clean (no violation).
    pub clean: bool,
    /// Violation tags recorded against the episode, if any.
    pub violations: Vec<String>,
    /// Refusal reasons the router recorded for this decision's actions.
    pub refusals: Vec<String>,
}

impl Observation {
    /// A prediction is *miscalibrated high* when the agent was confident and
    /// wrong. The single most actionable calibration signal, because it is
    /// the one that costs the most under the Brier term.
    pub fn overconfident_miss(&self) -> bool {
        self.stated_confidence >= 0.8 && !self.resolved
    }

    /// Confident and right — the case that *earns* the confidence.
    pub fn confident_hit(&self) -> bool {
        self.stated_confidence >= 0.8 && self.resolved
    }
}

/// Everything the miner needs, distilled from a ledger.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Corpus {
    pub observations: Vec<Observation>,
    /// Every breach the ledger recorded, by tag, with a count. Breaches are
    /// the sharpest learning signal there is: the agent reached for something
    /// it was refused, and did so more than once.
    pub breach_counts: HashMap<String, u64>,
}

impl Corpus {
    /// Distil a ledger into the miner's view.
    ///
    /// Deliberately tolerant of a partial ledger: a decision with no outcome
    /// row yet is simply not an observation, rather than an error. A run in
    /// progress must be mineable, not just a finished one.
    pub fn from_ledger(ledger: &Ledger) -> Result<Self, LedgerError> {
        // Stated confidence, indexed by decision id, from the record written
        // before the action.
        let mut confidence: HashMap<String, f64> = HashMap::new();
        for e in ledger.by_kind("decision_recorded")? {
            if let Event::DecisionRecorded { record } = &e.event {
                confidence.insert(
                    record.id().to_string(),
                    record.prediction().confidence.get(),
                );
            }
        }

        // Oracle outcomes.
        let mut resolved: HashMap<String, bool> = HashMap::new();
        for e in ledger.by_kind("outcome_observed")? {
            if let Event::OutcomeObserved {
                decision, resolved: r, ..
            } = &e.event
            {
                resolved.insert(decision.to_string(), *r);
            }
        }

        // Episode scores → clean/violated + violation tags.
        let mut clean: HashMap<String, bool> = HashMap::new();
        let mut violations: HashMap<String, Vec<String>> = HashMap::new();
        for e in ledger.by_kind("episode_scored")? {
            if let Event::EpisodeScored { decision, utility } = &e.event {
                clean.insert(decision.to_string(), utility.is_clean());
                if let samaritan_kernel::EpisodeUtility::Violated { violations: v } = utility {
                    violations.insert(
                        decision.to_string(),
                        v.iter().map(|x| x.tag.clone()).collect(),
                    );
                }
            }
        }

        // Router refusals, per decision.
        let mut refusals: HashMap<String, Vec<String>> = HashMap::new();
        for e in ledger.by_kind("mutation_refused")? {
            if let Event::MutationRefused { refusal, .. } = &e.event {
                // Not decision-scoped, but a run-level signal; bucketed under
                // a shared key so a recurring refusal is still counted.
                refusals
                    .entry("__run__".to_string())
                    .or_default()
                    .push(refusal.to_string());
            }
        }

        // Breach counts, the sharpest signal.
        let mut breach_counts: HashMap<String, u64> = HashMap::new();
        for e in ledger.breaches()? {
            if let Event::MutationRefused { refusal, .. } = &e.event {
                *breach_counts.entry(refusal.to_string()).or_insert(0) += 1;
            }
        }

        // Join into observations for every decision that has an outcome.
        let mut observations = Vec::new();
        for (id, r) in &resolved {
            observations.push(Observation {
                decision: id.clone(),
                stated_confidence: confidence.get(id).copied().unwrap_or(0.5),
                resolved: *r,
                clean: clean.get(id).copied().unwrap_or(true),
                violations: violations.get(id).cloned().unwrap_or_default(),
                refusals: refusals.get("__run__").cloned().unwrap_or_default(),
            });
        }
        // Stable order, so a mined lesson set is reproducible from a ledger.
        observations.sort_by(|a, b| a.decision.cmp(&b.decision));

        Ok(Self {
            observations,
            breach_counts,
        })
    }

    /// Build from a batch's `(confidence, resolved)` predictions, taking the
    /// breach counts from `ledger`.
    ///
    /// This is the runner's path: calibration is mined from the episodes just
    /// run (which the runner holds in hand), while breaches — which accumulate
    /// across the whole run and are not per-episode — come from the durable
    /// record. Reading calibration from the batch rather than re-reading the
    /// ledger avoids re-mining the same rows every round and keeps mining
    /// independent of whether the episode layer chose to log per-decision.
    pub fn from_predictions(
        predictions: &[(f64, bool)],
        ledger: &Ledger,
    ) -> Result<Self, LedgerError> {
        let mut breach_counts: HashMap<String, u64> = HashMap::new();
        for e in ledger.breaches()? {
            if let Event::MutationRefused { refusal, .. } = &e.event {
                *breach_counts.entry(refusal.to_string()).or_insert(0) += 1;
            }
        }
        let observations = predictions
            .iter()
            .enumerate()
            .map(|(i, (confidence, resolved))| Observation {
                decision: format!("batch:{i}"),
                stated_confidence: *confidence,
                resolved: *resolved,
                clean: true,
                violations: Vec::new(),
                refusals: Vec::new(),
            })
            .collect();
        Ok(Self {
            observations,
            breach_counts,
        })
    }

    pub fn len(&self) -> usize {
        self.observations.len()
    }

    pub fn is_empty(&self) -> bool {
        self.observations.is_empty()
    }
}
