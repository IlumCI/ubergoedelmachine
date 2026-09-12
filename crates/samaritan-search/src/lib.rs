//! Nested Rollout Policy Adaptation, recursing over depth of self-reference.
//!
//! [Rosin's NRPA](https://www.ijcai.org/Proceedings/11/Papers/115.pdf)
//! recurses over search levels: level *n* runs level *n*−1 repeatedly and
//! adapts a rollout policy toward whatever the level below found. Every
//! published application nests over search quality on a fixed task. Here the
//! nesting parameter is **meta-level of self-reference**:
//!
//! | level | a rollout is | what it mutates |
//! |---|---|---|
//! | 0 | one batch of task episodes | nothing — this is the playout |
//! | 1 | a level-0 batch | lessons, prompt fragments, router knobs |
//! | 2 | a level-1 search | the mutation grammar, adapt rate, budget split |
//! | 3 | a level-2 search | the search code itself |
//!
//! # Why NRPA and not some other optimiser
//!
//! Rosin passes the policy **by value** into the recursive call, and the
//! paper is explicit that level *n*−1's final policy is not returned to level
//! *n* — only `(score, sequence)` propagates upward. Adaptations made inside
//! a recursive call die when it returns.
//!
//! That is exactly the containment a self-modifying stack needs, and it is
//! not a safety feature bolted on: it is what the algorithm already does.
//! Every meta-level edit is speculative and rolled back unless it earns
//! promotion by improving the score that propagates up. [`MutationPolicy`] is
//! plain owned data with no interior mutability precisely so that `clone()`
//! means what this relies on it meaning.
//!
//! # Two details that are easy to get wrong
//!
//! **Acceptance is `>=`, not `>`.** Ties replace the incumbent best
//! sequence. Rosin reports this "preserves the property of return to the best
//! score found so far, while aiding exploration", and that strict improvement
//! performed substantially worse. It looks like a typo and is not.
//!
//! **`adapt` is out of place.** The softmax normaliser and the subtracted
//! probabilities are computed from the *old* policy while updates accumulate
//! into the copy — one batched gradient step, not a sequence of in-place
//! updates. See [`MutationPolicy::adapt`].
//!
//! Nothing here commits anything. A search returns a *candidate*; whether it
//! may be kept is `samaritan-cert`'s decision.

pub mod state;

use samaritan_dsl::{Mutation, MutationCode, MutationPolicy};
use serde::{Deserialize, Serialize};

pub use state::{Grammar, GrammarOp, Lesson, PolicyState};

/// What a rollout chose, and what it could have chosen at each step.
///
/// The legal set is carried because [`MutationPolicy::adapt`] needs it: the
/// gradient pushes weight toward the move taken and away from the
/// alternatives that were available at that moment, which cannot be
/// reconstructed afterwards.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Trace {
    pub steps: Vec<(MutationCode, Vec<MutationCode>)>,
    pub mutations: Vec<Mutation>,
}

/// The problem the search is searching.
///
/// Separating this from the algorithm is what lets the NRPA implementation be
/// tested against a benchmark with a known answer, rather than only against a
/// language model whose behaviour is itself the uncertainty.
pub trait Domain {
    /// Mutation classes available from this state.
    ///
    /// The coding is the domain's to choose. Rosin's `code(node, i)` is
    /// context-dependent, and a domain that wants position-specific learning
    /// should encode position into the code; one that wants to generalise
    /// across contexts — as self-modification does, since there are
    /// unboundedly many lesson texts but few kinds of move — should not.
    fn legal(&self, state: &PolicyState) -> Vec<MutationCode>;

    /// Turn a chosen class into a concrete mutation. `None` abandons the step.
    fn instantiate(&mut self, code: &MutationCode, state: &PolicyState) -> Option<Mutation>;

    /// Apply a mutation to a scratch state.
    fn apply(&mut self, state: &mut PolicyState, m: &Mutation) -> bool;

    /// Score a finished state. Higher is better.
    fn evaluate(&mut self, state: &PolicyState) -> f64;

    /// Mutations per rollout.
    fn depth(&self) -> usize;

    /// A number in `[0, 1)`. Supplied by the domain so a search is
    /// reproducible from a seed the caller controls.
    fn random(&mut self) -> f64;
}

/// Per-level search parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Budget {
    /// Iterations at each level, index 0 unused.
    ///
    /// Cost is the product of these, which is why level 3 is a weekly event
    /// and level 1 runs continuously. Rosin used 100 at every level.
    pub iterations: Vec<u32>,
    /// Adapt step size. Rosin used 1.0 throughout.
    pub alpha: f64,
}

impl Default for Budget {
    fn default() -> Self {
        Self {
            iterations: vec![0, 100, 40, 10],
            alpha: 1.0,
        }
    }
}

impl Budget {
    fn iters(&self, level: u8) -> u32 {
        self.iterations
            .get(level as usize)
            .copied()
            .unwrap_or(10)
            .max(1)
    }
}

/// Sample one rollout: a chain of mutations drawn from the policy, applied to
/// a scratch copy of `base`, then scored.
pub fn playout<D: Domain>(
    domain: &mut D,
    policy: &MutationPolicy,
    base: &PolicyState,
) -> (f64, Trace) {
    let mut state = base.clone();
    let mut trace = Trace::default();

    for _ in 0..domain.depth() {
        let legal = domain.legal(&state);
        if legal.is_empty() {
            break;
        }

        let probs = policy.distribution(&legal);
        let r = domain.random();
        let mut acc = 0.0;
        let mut chosen = legal.len() - 1;
        for (i, p) in probs.iter().enumerate() {
            acc += p;
            if r < acc {
                chosen = i;
                break;
            }
        }
        let code = legal[chosen].clone();

        let Some(m) = domain.instantiate(&code, &state) else {
            continue;
        };
        if !domain.apply(&mut state, &m) {
            // A refused mutation is a dead end for this step, not for the
            // rollout: the search should learn that the move does not pay,
            // which it only can if the step is recorded.
            trace.steps.push((code, legal));
            continue;
        }
        trace.steps.push((code, legal));
        trace.mutations.push(m);
    }

    (domain.evaluate(&state), trace)
}

/// Nested rollout policy adaptation over self-reference depth.
///
/// Written recursively and level-generic from the start, even though only
/// `level = 1` is called today. A `level == 0` fast path is the one change
/// that would make adding level 2 a rewrite, so it does not get written.
pub fn nrsi<D: Domain>(
    level: u8,
    policy: &MutationPolicy,
    domain: &mut D,
    base: &PolicyState,
    budget: &Budget,
) -> (f64, Trace) {
    if level == 0 {
        return playout(domain, policy, base);
    }

    let mut best = f64::NEG_INFINITY;
    let mut best_trace = Trace::default();

    // The copy that makes the whole thing safe. Adaptations below this line
    // are invisible to the caller once this function returns.
    let mut p = policy.clone();

    for _ in 0..budget.iters(level) {
        let (score, trace) = nrsi(level - 1, &p, domain, base, budget);

        // `>=`, deliberately. See the module docs.
        if score >= best {
            best = score;
            best_trace = trace;
        }
        p = p.adapt(&best_trace.steps, budget.alpha);
    }

    (best, best_trace)
}

/// A finished search: the candidate, and what it scored on the tasks the
/// search itself used.
///
/// That score is *not* evidence the candidate is better. It is the number the
/// search optimised, computed on the data it optimised against, and treating
/// it as evidence is precisely the circularity `samaritan-cert` refuses. The
/// candidate has to be re-run on held-out tasks before anything is committed.
#[derive(Debug, Clone, PartialEq)]
pub struct Candidate {
    pub mutations: Vec<Mutation>,
    pub search_score: f64,
}

/// Run a level-1 search and return the candidate it found.
pub fn propose<D: Domain>(
    domain: &mut D,
    base: &PolicyState,
    budget: &Budget,
) -> Candidate {
    let (score, trace) = nrsi(1, &MutationPolicy::new(), domain, base, budget);
    Candidate {
        mutations: trace.mutations,
        search_score: score,
    }
}
