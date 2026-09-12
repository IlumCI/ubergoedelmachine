//! Watch the machine try to patch its own source, and see the gate hold.
//!
//!     cargo run -p samaritan-run --example propose_patch
//!
//! The level-3 loop end to end: the local model reads one of the search's own
//! files and a goal, writes a unified diff, and that diff runs the full gate —
//! screened against the frozen paths, built and tested in a throwaway git
//! worktree, and (here) put to a human on the terminal.
//!
//! Nothing is ever written to the live tree by this example. A patch that
//! clears every gate produces an `AcceptedPatch` and is printed; applying it is
//! a separate, deliberate act, not something a demo does unattended.
//!
//! Env: SAMARITAN_URL (model server), TARGET (file to patch; default the
//! search's lib.rs), GOAL, SEED.

use std::time::Duration;

use samaritan_agent::{Agent, AgentConfig, Constrain};
use samaritan_kernel::{Admission, Approval, ApprovalGate, ApprovalRequest, GateError};
use samaritan_run::patch::WorktreeVerifier;
use samaritan_run::propose::GenerativePatchProposer;
use samaritan_dsl::Mutation;
use samaritan_search::{gate_code_patch, PatchContext, PatchProposer};

/// The fallback: propose nothing. A demo should show the model's own work or
/// visibly decline, never a canned patch dressed up as the model's.
struct NoFallback;
impl PatchProposer for NoFallback {
    fn propose(&mut self, _ctx: &PatchContext) -> Option<Mutation> {
        None
    }
}

/// Asks on the terminal. Reads a single line; anything but "y" is a refusal,
/// and end-of-input is a refusal too — the safe default when no one is there.
struct TerminalGate;
impl ApprovalGate for TerminalGate {
    fn ask(&mut self, _req: &ApprovalRequest) -> Result<Approval, GateError> {
        use std::io::Write;
        print!("\nApply this patch to the tree? [y/N] ");
        let _ = std::io::stdout().flush();
        let mut line = String::new();
        match std::io::stdin().read_line(&mut line) {
            Ok(0) => Ok(Approval::Refuse), // EOF: nobody home
            Ok(_) => Ok(if line.trim().eq_ignore_ascii_case("y") {
                Approval::Allow
            } else {
                Approval::Refuse
            }),
            Err(e) => Err(GateError::Channel(e.to_string())),
        }
    }
}

fn main() {
    let base_url =
        std::env::var("SAMARITAN_URL").unwrap_or_else(|_| "http://127.0.0.1:8080/v1".into());
    let target = std::env::var("TARGET")
        .unwrap_or_else(|_| "crates/samaritan-search/src/lib.rs".into());
    let goal = std::env::var("GOAL").unwrap_or_else(|_| {
        "add a brief doc comment or a small, clearly-correct clarity improvement".into()
    });
    let seed = std::env::var("SEED").ok().and_then(|s| s.parse().ok()).unwrap_or(1);

    let repo = std::env::current_dir().expect("cwd");
    let source = match std::fs::read_to_string(repo.join(&target)) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("could not read {target}: {e}");
            std::process::exit(1);
        }
    };
    let ctx = PatchContext { goal: goal.clone(), target_path: target.clone(), current_source: source };
    if ctx.targets_frozen() {
        eprintln!("{target} is frozen; the gate would refuse any patch to it. Pick a mutable file.");
        std::process::exit(2);
    }

    let agent = Agent::new(AgentConfig {
        base_url,
        model: "samaritan-playout".into(),
        temperature: 0.4, // a patch wants care, not creativity
        max_tokens: 1536,
        repeat_penalty: 1.15,
        constrain: Constrain::None,
        timeout: Duration::from_secs(900),
        max_retries: 1,
        seed: Some(seed),
        ..Default::default()
    });
    let mut proposer = GenerativePatchProposer::new(agent, 0.4, seed, NoFallback);

    println!("goal:   {goal}");
    println!("target: {target}\n");

    let Some(mutation) = proposer.propose(&ctx) else {
        if let Some(err) = &proposer.last_fell_back {
            println!("the model produced no usable diff ({err}).");
        } else {
            println!("the proposer offered nothing.");
        }
        return;
    };
    if let Mutation::CodePatch { unified_diff } = &mutation {
        println!("── the model's proposed patch ──────────────────\n{unified_diff}\n");
    }

    let scratch = std::env::temp_dir().join("samaritan-l3-verify");
    let mut verifier = WorktreeVerifier::new(&repo, scratch);
    let admission = Admission::new(3);

    // The gate needs the human's answer. Ask only once it builds and passes, so
    // run the verify-gated flow: screen + build + test happen inside
    // gate_code_patch, and it returns AwaitingHuman-shaped refusals before we
    // bother a person. Here we pass the terminal answer directly.
    let mut gate = TerminalGate;
    // Two-pass: first with a refusal to learn whether it even reaches the human,
    // so we do not prompt for a patch that fails to build.
    match gate_code_patch(&mutation, &admission, &mut verifier, Approval::Refuse) {
        Err(samaritan_search::PatchRefusal::HumanRefused) => {
            // It built and passed; now actually ask.
            use samaritan_kernel::Tier;
            let action = probe_action();
            let req = ApprovalRequest {
                action: &action,
                rationale: &goal,
                prediction: &probe_prediction(),
                tier: Tier::Confirm,
            };
            let answer = gate.ask(&req).unwrap_or(Approval::Refuse);
            match gate_code_patch(&mutation, &admission, &mut verifier, answer) {
                Ok(accepted) => {
                    println!("\nACCEPTED. It builds, passes, and was approved.");
                    println!("{}", accepted.report.detail);
                    println!("\n(Not applied — applying is a separate, deliberate step.)");
                }
                Err(r) => println!("\nrefused: {r}"),
            }
        }
        Err(other) => println!("refused before reaching a human: {other}"),
        Ok(_) => unreachable!("a refusal was passed"),
    }
}

// Minimal stand-ins so the ApprovalRequest can be rendered; the patch, not the
// action, is what matters here.
fn probe_action() -> samaritan_dsl::ProposedAction {
    use samaritan_dsl::{ActionKind, BlastRadius, ProposedAction, Reversibility};
    ProposedAction {
        kind: ActionKind::Write,
        reversibility: Reversibility::Snapshot,
        blast_radius: BlastRadius::Repo,
        intent: "level-3 self-patch".into(),
        payload: serde_json::Value::Null,
    }
}

fn probe_prediction() -> samaritan_dsl::Prediction {
    samaritan_dsl::Prediction {
        outcome: "the search's own source is improved".into(),
        confidence: samaritan_dsl::Confidence::new(0.5).unwrap(),
    }
}
