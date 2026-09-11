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
/// Field order is fixed and matches the struct, which matters beyond
/// tidiness: a fixed order means the model never spends tokens deciding what
/// to emit next, and the shared prefix of every response stays identical for
/// as long as possible.
///
/// Note what the grammar does *not* let the model produce: no `id`, no
/// `policy_version`, no `authority`. Those are the harness's to set. A model
/// that could declare its own provenance could declare it trustworthy.
pub const DRAFT_DECISION_GBNF: &str = r##"
root ::= "{" ws
  "\"situation\":" ws string ws "," ws
  "\"options\":" ws options ws "," ws
  "\"chosen\":" ws index ws "," ws
  "\"rationale\":" ws string ws "," ws
  "\"prediction\":" ws prediction ws "," ws
  "\"actions\":" ws actions ws
"}" ws

options ::= "[" ws option (ws "," ws option){1,4} ws "]"
option ::= "{" ws
  "\"summary\":" ws string ws "," ws
  "\"assessment\":" ws string ws
"}"

prediction ::= "{" ws
  "\"outcome\":" ws string ws "," ws
  "\"confidence\":" ws confidence ws
"}"

actions ::= "[" ws (action (ws "," ws action){0,7} ws)? "]"
action ::= "{" ws
  "\"kind\":" ws kind ws "," ws
  "\"reversibility\":" ws reversibility ws "," ws
  "\"blast_radius\":" ws blast ws "," ws
  "\"intent\":" ws string ws "," ws
  "\"payload\":" ws payload ws
"}"

kind ::= "\"read\"" | "\"write\"" | "\"exec\"" | "\"net\"" | "\"git_history\""
reversibility ::= "\"trivial\"" | "\"snapshot\"" | "\"costly\"" | "\"irreversible\""
blast ::= "\"episode\"" | "\"repo\"" | "\"machine\"" | "\"external\""

payload ::= read-file | list-dir | write-file | delete-file | run
read-file   ::= "{" ws "\"do\":" ws "\"read_file\"" ws "," ws "\"path\":" ws string ws "}"
list-dir    ::= "{" ws "\"do\":" ws "\"list_dir\"" ws "," ws "\"path\":" ws string ws "}"
write-file  ::= "{" ws "\"do\":" ws "\"write_file\"" ws "," ws "\"path\":" ws string ws "," ws "\"contents\":" ws string ws "}"
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
"##;

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
