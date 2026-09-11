//! What the machine is allowed to change about itself.
//!
//! A Gödel machine that can rewrite its own leash will rewrite its own leash.
//! Every mutation is screened here before it is evaluated, not after, so that a
//! modification which would weaken the gate never gets a score attached to it
//! and never becomes something the search is pulled toward.

use samaritan_dsl::{Knob, Mutation};
use serde::{Deserialize, Serialize};

/// Path prefixes the machine may never patch.
///
/// The certificate checker is on this list alongside the gate itself: a system
/// that can rewrite the test it must pass has no meaningful test.
pub const FROZEN_PATHS: &[&str] = &[
    "crates/samaritan-kernel/",
    "crates/samaritan-cert/",
    "Cargo.toml",
    "Cargo.lock",
    ".cargo/",
    "rust-toolchain.toml",
];

/// Why a mutation was refused.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Refusal {
    /// The mutation belongs to a self-reference level that is not switched on.
    LevelNotEnabled { level: u8, max_enabled: u8 },
    /// The mutation would edit something in [`FROZEN_PATHS`].
    ///
    /// Logged as an attempted breach rather than an ordinary rejection.
    TouchesFrozenCore { path: String },
    /// A knob was pushed outside the range a human fixed.
    KnobOutOfRange { knob: Knob, value: f64, lo: f64, hi: f64 },
    /// A knob value that is not a number at all.
    KnobNotFinite { knob: Knob, value: f64 },
    /// A diff that could not be parsed well enough to know what it touches.
    ///
    /// Refused rather than guessed at: an unreadable patch is exactly the shape
    /// a frozen-core edit would take if it wanted through.
    Unreadable { detail: String },
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Refusal::LevelNotEnabled { level, max_enabled } => {
                write!(f, "level {level} mutations are not enabled (max {max_enabled})")
            }
            Refusal::TouchesFrozenCore { path } => {
                write!(f, "mutation targets frozen path {path}")
            }
            Refusal::KnobOutOfRange { knob, value, lo, hi } => {
                write!(f, "{knob:?} = {value} is outside [{lo}, {hi}]")
            }
            Refusal::KnobNotFinite { knob, value } => write!(f, "{knob:?} = {value} is not finite"),
            Refusal::Unreadable { detail } => write!(f, "unreadable mutation: {detail}"),
        }
    }
}

/// Whether a refusal should be treated as an attempted breach of the core.
///
/// Ordinary refusals are search feedback. Breaches are an incident: they mean
/// the optimiser found the constraint and pushed on it.
pub fn is_breach(r: &Refusal) -> bool {
    matches!(r, Refusal::TouchesFrozenCore { .. })
}

/// The frozen range for a knob.
///
/// These bound what a human can even opt into. `CalibrationCeiling` is the one
/// to read closely: it caps how badly calibrated the agent may be while still
/// earning promotions, so allowing it to rise would let the machine buy
/// autonomy by lowering its own standard for deserving it.
pub fn knob_range(knob: Knob) -> (f64, f64) {
    match knob {
        Knob::PromotionStreak => (3.0, 100.0),
        Knob::CalibrationCeiling => (0.0, 0.35),
        Knob::LessonBudget => (0.0, 64.0),
        Knob::DecisionTemperature => (0.0, 1.5),
    }
}

/// Frozen admission policy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Admission {
    /// Highest self-reference level currently switched on.
    ///
    /// v1 ships at 1. Levels 2 and 3 exist in the grammar so that turning them
    /// on later is a config change rather than a redesign, and turning them on
    /// is a human act.
    max_enabled_level: u8,
}

impl Default for Admission {
    fn default() -> Self {
        Self {
            max_enabled_level: 1,
        }
    }
}

impl Admission {
    pub fn new(max_enabled_level: u8) -> Self {
        Self { max_enabled_level }
    }

    pub fn max_enabled_level(&self) -> u8 {
        self.max_enabled_level
    }

    /// Screen one mutation.
    ///
    /// Out-of-range knobs are refused rather than silently clamped. Clamping
    /// would hand the search a proposal it did not make and hide the fact that
    /// it is pressing against a bound, which is exactly the signal a human
    /// reading the ledger wants to see.
    pub fn admit(&self, m: &Mutation) -> Result<(), Refusal> {
        let level = m.level();
        if level > self.max_enabled_level {
            return Err(Refusal::LevelNotEnabled {
                level,
                max_enabled: self.max_enabled_level,
            });
        }

        match m {
            Mutation::ThresholdSet { knob, value } => {
                if !value.is_finite() {
                    return Err(Refusal::KnobNotFinite {
                        knob: *knob,
                        value: *value,
                    });
                }
                let (lo, hi) = knob_range(*knob);
                if *value < lo || *value > hi {
                    return Err(Refusal::KnobOutOfRange {
                        knob: *knob,
                        value: *value,
                        lo,
                        hi,
                    });
                }
                Ok(())
            }
            Mutation::CodePatch { unified_diff } => screen_diff(unified_diff),
            _ => Ok(()),
        }
    }
}

/// Reject a diff that touches anything frozen.
///
/// Deliberately conservative. Every `+++`/`---` header is checked, path
/// separators are normalised, and a diff with no recognisable file headers is
/// refused rather than admitted, because "I could not tell what this changes"
/// must never resolve to "so let it through".
fn screen_diff(diff: &str) -> Result<(), Refusal> {
    let mut saw_header = false;

    for line in diff.lines() {
        let rest = match line.strip_prefix("--- ").or_else(|| line.strip_prefix("+++ ")) {
            Some(r) => r,
            None => continue,
        };
        saw_header = true;

        // Strip the a/ or b/ prefix git puts on diff headers, and any tab-separated
        // timestamp that follows the path.
        let path = rest
            .split('\t')
            .next()
            .unwrap_or(rest)
            .trim()
            .replace('\\', "/");
        let path = path
            .strip_prefix("a/")
            .or_else(|| path.strip_prefix("b/"))
            .unwrap_or(&path)
            .to_string();

        if path == "/dev/null" {
            continue;
        }

        // Path traversal would let a diff reach a frozen path by a name that
        // does not literally start with one.
        if path.split('/').any(|seg| seg == "..") {
            return Err(Refusal::Unreadable {
                detail: format!("path escapes the workspace: {path}"),
            });
        }

        if let Some(frozen) = FROZEN_PATHS.iter().find(|f| path.starts_with(**f)) {
            let _ = frozen;
            return Err(Refusal::TouchesFrozenCore { path });
        }
    }

    if !saw_header {
        return Err(Refusal::Unreadable {
            detail: "no file headers found in diff".to_string(),
        });
    }
    Ok(())
}
