//! Samaritan's decision language.
//!
//! Two vocabularies live here and nothing else:
//!
//! - [`decision`] — what the model is allowed to say. It reports a situation,
//!   the options it weighed, what it chose, what it expects to happen and how
//!   sure it is, and what it wants done. It does not get to say how dangerous
//!   any of that is; the frozen kernel decides that from the facts reported.
//! - [`mutation`] — how the machine is allowed to change itself, and the
//!   policy the nested search adapts over those changes.
//!
//! This crate is deliberately inert: types, validation, and pure functions. No
//! I/O, no randomness, no execution. Everything that can actually do something
//! lives downstream of a gate.

pub mod decision;
pub mod error;
pub mod hash;
pub mod mutation;

pub use decision::{
    ActionKind, Authority, BlastRadius, Confidence, DecisionId, DecisionOption, DecisionRecord,
    PolicyVersion, Prediction, ProposedAction, Reversibility,
};
pub use error::DslError;
pub use hash::{Digest, hash_json};
pub use mutation::{Knob, LessonId, Mutation, MutationCode, MutationPolicy};

/// Compile-time guard for the property NRPA depends on.
///
/// `RefUnwindSafe` fails for `Cell` and `RefCell`, so this catches the easy
/// ways to accidentally give the policy shared mutable state. It does not catch
/// `Arc<Mutex<_>>`; the behavioural test `clone_is_deeply_independent` covers
/// that case, and the two together are the actual guarantee.
const _: fn() = || {
    fn assert_value_semantics<T: Clone + Send + Sync + std::panic::RefUnwindSafe>() {}
    assert_value_semantics::<MutationPolicy>();
};
