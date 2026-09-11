//! Errors raised when a decision record or mutation fails validation.
//!
//! Validation is not advisory. Records arrive from a language model, which means
//! every invariant this crate claims has to be enforced at the boundary rather
//! than assumed.

use thiserror::Error;

#[derive(Debug, Error, PartialEq)]
pub enum DslError {
    #[error("a decision needs at least two options to be a decision; got {0}")]
    TooFewOptions(usize),

    #[error("chosen option index {chosen} is out of range for {count} options")]
    ChosenOutOfRange { chosen: usize, count: usize },

    #[error("confidence must be within [0, 1]; got {0}")]
    ConfidenceOutOfRange(f64),

    #[error("confidence must be a real number; got {0}")]
    ConfidenceNotFinite(f64),

    #[error("{field} must not be blank")]
    Blank { field: &'static str },

    #[error("weight must be finite and positive; got {0}")]
    BadWeight(f64),
}
