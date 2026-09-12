//! The level-3 proposer: the model writing patches to the search's own source.
//!
//! This is the payload of the Gödelian move. Levels 1 and 2 rewrite *data* the
//! search reads — lessons, thresholds, the grammar. Level 3 rewrites the
//! *code*, and the only thing that can author a code change the designers did
//! not foresee is the model itself. So this asks it to.
//!
//! Every safety property lives downstream, in
//! [`samaritan_search::gate_code_patch`]: whatever the model emits is screened
//! against the frozen paths, built and tested in a throwaway worktree, and put
//! to a human, in that order. That is deliberate. A generator is the one part
//! of a self-modifying system you cannot make safe by construction — it can
//! propose anything — so it is made *harmless* by construction instead: it
//! proposes, and something it cannot influence disposes. Nothing here writes to
//! the tree; nothing here even decides. It returns a candidate and stops.
//!
//! The model is not asked for JSON. A unified diff is not a grammar a sampler
//! can be constrained into, so this takes free-form output and lifts the diff
//! out of it — tolerant of the fences and preamble a chat model wraps around
//! code, strict about what a diff has to look like.

use samaritan_agent::{Agent, AgentError};
use samaritan_dsl::Mutation;
use samaritan_search::{PatchContext, PatchProposer};

/// The instructions the proposer runs under. Honest about what it is doing —
/// improving its own machinery — and blunt about the two lines it must not
/// cross, because a proposal that crosses them only wastes a build.
pub const PROPOSER_SYSTEM: &str = "\
You are improving the source code of a self-improving agent's own search. You \
are given one Rust file and a goal. Emit a single unified diff (git format, \
with `--- a/<path>` and `+++ b/<path>` headers and `@@` hunks) that changes \
THAT FILE ONLY to advance the goal.

Hard rules, because a diff that breaks them is thrown away unbuilt:
- Patch only the file you were given. Do not touch other files, and never touch \
  crates/samaritan-kernel/ or crates/samaritan-cert/ — those are frozen.
- Keep it minimal and buildable. A small correct change that compiles and keeps \
  the tests green beats an ambitious one that does not.
- Do not delete or weaken tests to make a change pass.

Output only the diff. No explanation, no prose, no fences.";

/// A [`PatchProposer`] backed by the local model, with a fallback for when the
/// model returns nothing usable.
///
/// `F` is a fallback proposer, used when the model errors or emits something
/// that is not a diff — mirroring the Deviant's opening book, and for the same
/// reason: a generator that silently stalls the loop is worse than one that
/// visibly falls back. `last_fell_back` records when that happened.
pub struct GenerativePatchProposer<F: PatchProposer> {
    agent: Agent,
    temperature: f64,
    seed: u64,
    round: u64,
    fallback: F,
    pub last_fell_back: Option<String>,
}

impl<F: PatchProposer> GenerativePatchProposer<F> {
    pub fn new(agent: Agent, temperature: f64, seed: u64, fallback: F) -> Self {
        Self { agent, temperature, seed, round: 0, fallback, last_fell_back: None }
    }

    fn user_prompt(ctx: &PatchContext) -> String {
        format!(
            "Goal: {}\n\nFile to patch: {}\n\n--- current contents of {} ---\n{}\n--- end ---\n\n\
             Emit the unified diff now.",
            ctx.goal, ctx.target_path, ctx.target_path, ctx.current_source
        )
    }

    fn try_generate(&mut self, ctx: &PatchContext) -> Result<Mutation, AgentError> {
        let (content, _usage) = self.agent.complete(
            PROPOSER_SYSTEM,
            &Self::user_prompt(ctx),
            None,
            None,
            self.temperature,
            self.seed.wrapping_add(self.round),
        )?;
        let diff = extract_unified_diff(&content).ok_or_else(|| {
            AgentError::Parse(format!("no unified diff in the reply; got: {content}"))
        })?;
        Ok(Mutation::CodePatch { unified_diff: diff })
    }
}

impl<F: PatchProposer> PatchProposer for GenerativePatchProposer<F> {
    fn propose(&mut self, ctx: &PatchContext) -> Option<Mutation> {
        self.round += 1;
        match self.try_generate(ctx) {
            Ok(m) => {
                self.last_fell_back = None;
                Some(m)
            }
            Err(e) => {
                self.last_fell_back = Some(e.to_string());
                self.fallback.propose(ctx)
            }
        }
    }
}

/// Lift a unified diff out of a chat model's reply.
///
/// A chat model wraps code in ```fences and preamble; a diff, though, has a
/// shape — it begins at the first `--- ` header and every subsequent line is
/// part of it until a fence or the end. This keeps from that first header to
/// the end, drops a trailing fence, and returns `None` if there is no header or
/// no `+++`/`@@` to go with it, so a reply that merely talks about a diff is not
/// mistaken for one.
pub fn extract_unified_diff(content: &str) -> Option<String> {
    let start = content.find("--- ")?;
    let mut body = &content[start..];
    // Cut a closing code fence if the model wrapped the diff in one.
    if let Some(fence) = body.find("\n```") {
        body = &body[..fence];
    }
    let body = body.trim_end();
    // A real diff needs a target header and at least one hunk or old/new pair.
    if !body.contains("+++ ") || !(body.contains("@@") || body.contains("\n+") || body.contains("\n-")) {
        return None;
    }
    Some(format!("{body}\n"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_clean_diff_is_extracted_verbatim() {
        let reply = "--- a/x.rs\n+++ b/x.rs\n@@ -1 +1 @@\n-a\n+b\n";
        let got = extract_unified_diff(reply).expect("a diff");
        assert!(got.contains("--- a/x.rs") && got.contains("+b"));
    }

    #[test]
    fn a_diff_is_lifted_out_of_fences_and_preamble() {
        let reply = "Sure, here is the change:\n\n```diff\n--- a/x.rs\n+++ b/x.rs\n@@ -1 +1 @@\n-a\n+b\n```\nHope that helps!";
        let got = extract_unified_diff(reply).expect("a diff");
        assert!(got.starts_with("--- a/x.rs"), "{got}");
        assert!(!got.contains("Hope that helps"), "trailing prose must be dropped");
        assert!(!got.contains("```"), "the fence must be dropped");
    }

    #[test]
    fn prose_that_merely_mentions_a_diff_is_not_a_diff() {
        assert!(extract_unified_diff("I would change the --- header but I won't show a diff").is_none());
        assert!(extract_unified_diff("no diff here at all").is_none());
    }
}
