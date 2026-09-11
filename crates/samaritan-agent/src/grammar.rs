//! A GBNF grammar constraining the model to emit a valid draft decision.
//!
//! This is not a nicety. Two things make it close to mandatory here:
//!
//! **The model is abliterated.** Refusal ablation is a blunt edit to the
//! residual stream, and it reliably costs some instruction-following along
//! with the refusals. A model that drifts out of JSON one time in ten turns a
//! 55-second generation into a 110-second one, and the retry is not free —
//! it is the most expensive thing in the loop.
//!
//! **Constrained decoding cannot produce an unparseable record.** With the
//! grammar attached, `serde_json` failure stops being a runtime concern and
//! becomes a bug in this file. Structural validity is guaranteed by the
//! sampler; only the *semantic* invariants (at least two options, a chosen
//! index in range) still need checking, and those are already enforced by
//! [`samaritan_dsl::DecisionRecord::new`].
//!
//! llama.cpp applies this through the `grammar` field on a completion
//! request. Servers that do not support GBNF fall back to a JSON-schema
//! `response_format`, which is weaker — it constrains shape but not
//! enumerations as tightly — and weaker still is nothing at all, where the
//! agent has to parse-and-retry.

/// GBNF for [`crate::DraftDecision`].
///
/// **One rule per line, always.** llama.cpp's GBNF parser (verified against
/// build b10907) rejects a rule body that spans lines, and reports only
/// `failed to parse grammar` with no line number — so a wrapped rule costs an
/// afternoon of bisection. That is also why this is a raw string: written
/// with Rust escapes, the doubled backslashes needed for `\"` and `\\` become
/// unreadable, and an escaping mistake produces exactly the same
/// characterless error.
///
/// Field order is fixed and matches the struct, which matters beyond
/// tidiness: the model never spends tokens deciding what to emit next, and
/// the shared prefix of every response stays identical for as long as
/// possible.
///
/// Note what the grammar does *not* let the model produce: no `id`, no
/// `policy_version`, no `authority`. Those are the harness's to set. A model
/// that could declare its own provenance could declare it trustworthy.
pub const DRAFT_DECISION_GBNF: &str = r#"
root ::= "{" ws "\"situation\":" ws string ws "," ws "\"options\":" ws options ws "," ws "\"chosen\":" ws index ws "," ws "\"rationale\":" ws string ws "," ws "\"prediction\":" ws prediction ws "," ws "\"actions\":" ws actions ws "}" ws
options ::= "[" ws option (ws "," ws option){1,4} ws "]"
option ::= "{" ws "\"summary\":" ws string ws "," ws "\"assessment\":" ws string ws "}"
prediction ::= "{" ws "\"outcome\":" ws string ws "," ws "\"confidence\":" ws confidence ws "}"
actions ::= "[" ws (action (ws "," ws action){0,7} ws)? "]"
action ::= "{" ws "\"kind\":" ws kind ws "," ws "\"reversibility\":" ws reversibility ws "," ws "\"blast_radius\":" ws blast ws "," ws "\"intent\":" ws string ws "," ws "\"payload\":" ws payload ws "}"
kind ::= "\"read\"" | "\"write\"" | "\"exec\"" | "\"net\"" | "\"git_history\""
reversibility ::= "\"trivial\"" | "\"snapshot\"" | "\"costly\"" | "\"irreversible\""
blast ::= "\"episode\"" | "\"repo\"" | "\"machine\"" | "\"external\""
payload ::= read-file | list-dir | write-file | delete-file | run
read-file ::= "{" ws "\"do\":" ws "\"read_file\"" ws "," ws "\"path\":" ws string ws "}"
list-dir ::= "{" ws "\"do\":" ws "\"list_dir\"" ws "," ws "\"path\":" ws string ws "}"
write-file ::= "{" ws "\"do\":" ws "\"write_file\"" ws "," ws "\"path\":" ws string ws "," ws "\"contents\":" ws string ws "}"
delete-file ::= "{" ws "\"do\":" ws "\"delete_file\"" ws "," ws "\"path\":" ws string ws "}"
run ::= "{" ws "\"do\":" ws "\"run\"" ws "," ws "\"program\":" ws string ws "," ws "\"args\":" ws strings ws "," ws "\"timeout_secs\":" ws timeout ws "}"
strings ::= "[" ws (string (ws "," ws string){0,15} ws)? "]"
index ::= [0-4]
timeout ::= [1-9] [0-9]{0,3}
confidence ::= "0" | "1" | "0." [0-9]{1,3} | "1.0"
string ::= "\"" char* "\""
char ::= [^"\\\x00-\x1F] | "\\" (["\\bfnrt/] | "u" hex hex hex hex)
hex ::= [0-9a-fA-F]
ws ::= [ \t\n]*
"#;

/// JSON-schema equivalent, for servers that accept `response_format` but not
/// GBNF (LM Studio, vLLM, anything OpenAI-shaped).
///
/// Strictly weaker than the grammar — schema validation constrains the shape
/// but most servers enforce it by retry or by a looser decoder, so malformed
/// output is still reachable. Preferred over nothing; not preferred over
/// GBNF.
pub fn draft_decision_schema() -> serde_json::Value {
    use serde_json::json;

    let action = json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["kind", "reversibility", "blast_radius", "intent", "payload"],
        "properties": {
            "kind": {"enum": ["read", "write", "exec", "net", "git_history"]},
            "reversibility": {"enum": ["trivial", "snapshot", "costly", "irreversible"]},
            "blast_radius": {"enum": ["episode", "repo", "machine", "external"]},
            "intent": {"type": "string"},
            "payload": {"type": "object"}
        }
    });

    json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["situation", "options", "chosen", "rationale", "prediction", "actions"],
        "properties": {
            "situation": {"type": "string"},
            "options": {
                "type": "array",
                "minItems": 2,
                "maxItems": 5,
                "items": {
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["summary", "assessment"],
                    "properties": {
                        "summary": {"type": "string"},
                        "assessment": {"type": "string"}
                    }
                }
            },
            "chosen": {"type": "integer", "minimum": 0, "maximum": 4},
            "rationale": {"type": "string"},
            "prediction": {
                "type": "object",
                "additionalProperties": false,
                "required": ["outcome", "confidence"],
                "properties": {
                    "outcome": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0}
                }
            },
            "actions": {"type": "array", "maxItems": 8, "items": action}
        }
    })
}

#[cfg(test)]
mod tests {
    use super::DRAFT_DECISION_GBNF;

    /// The constraint that cost an afternoon, pinned so it cannot regress.
    ///
    /// A rule wrapped across lines produces `failed to parse grammar` from the
    /// server with no line number, which is close to undebuggable from the
    /// error alone.
    #[test]
    fn every_rule_is_on_one_line() {
        for line in DRAFT_DECISION_GBNF.lines() {
            let l = line.trim();
            if l.is_empty() {
                continue;
            }
            assert!(
                l.contains("::="),
                "continuation line found; llama.cpp needs one rule per line: {l:?}"
            );
        }
    }

    #[test]
    fn every_referenced_rule_is_defined() {
        let defined: Vec<&str> = DRAFT_DECISION_GBNF
            .lines()
            .filter_map(|l| l.split("::=").next())
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect();
        assert!(defined.contains(&"root"), "a grammar needs a root rule");

        for line in DRAFT_DECISION_GBNF.lines() {
            let Some(rhs) = line.split_once("::=").map(|(_, r)| r) else {
                continue;
            };
            // Identifiers outside quoted literals and character classes.
            let mut in_str = false;
            let mut in_class = false;
            let mut word = String::new();
            let mut chars = rhs.chars().peekable();
            while let Some(c) = chars.next() {
                match c {
                    '\\' if in_str || in_class => {
                        chars.next();
                    }
                    '"' => in_str = !in_str,
                    '[' if !in_str => in_class = true,
                    ']' if !in_str => in_class = false,
                    c if !in_str && !in_class && (c.is_ascii_alphanumeric() || c == '-' || c == '_') => {
                        word.push(c);
                        continue;
                    }
                    _ => {}
                }
                if !word.is_empty() {
                    if word.chars().next().is_some_and(|c| c.is_ascii_alphabetic()) {
                        assert!(
                            defined.contains(&word.as_str()),
                            "rule {word:?} is referenced but never defined"
                        );
                    }
                    word.clear();
                }
            }
        }
    }
}
