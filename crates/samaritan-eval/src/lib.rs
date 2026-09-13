//! Scoring the model on a held-out benchmark outside the arena — today,
//! Humanity's Last Exam.
//!
//! This is the *generalisation* probe. Everything else Samaritan measures is on
//! the repo surface it trains against; HLE is closed-form reasoning it has never
//! seen and cannot tune to, so a rising score is capability that transferred out
//! of the environment rather than a policy fitted to it. That is exactly the
//! question the three-arm experiment asks — does the arms race build general
//! ability, or only reward-hacking — and it is the capability half of the
//! public-repo milestone ([`samaritan_kernel::Capability`]).
//!
//! # What this ships, and the honest caveat on grading
//!
//! The scoring core here — [`load`], [`grade`], [`tally`] — is pure and tested;
//! the live model call lives in the `hle` example. Two caveats, stated so they
//! are not mistaken for more than they are:
//!
//! - **No dataset is bundled.** HLE is gated on its host and licensed; the
//!   operator supplies the JSONL, exactly as with the knowledge base and the
//!   task corpus. The format is documented on [`HleQuestion`].
//! - **Grading is a normalised match, not the official judge.** Real HLE grades
//!   free-form answers with an LLM judge. This does normalised string / token
//!   matching — cheap, deterministic, and good enough for a *directional*
//!   dashboard signal, but it will under-credit a correct answer phrased
//!   unusually. The recorded [`Event::HleEvaluated`] names the method, so the
//!   number is never mistaken for an official score.

use serde::{Deserialize, Serialize};

/// How an answer is checked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum AnswerType {
    /// A short free-form answer — a number, a name, a formula.
    ExactMatch,
    /// One option; the ground-truth answer is typically a letter.
    MultipleChoice,
}

impl Default for AnswerType {
    fn default() -> Self {
        AnswerType::ExactMatch
    }
}

/// One benchmark item, in HLE's shape. Extra fields in the source JSONL are
/// ignored, so a full HLE row deserializes without listing every column.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HleQuestion {
    #[serde(default)]
    pub id: String,
    pub question: String,
    pub answer: String,
    #[serde(default, rename = "answer_type", alias = "answerType")]
    pub answer_type: AnswerType,
}

/// Load questions from a JSONL blob — one object per line. Blank lines skipped;
/// a malformed line is an error naming its number, so a bad export fails loudly.
pub fn load(jsonl: &str) -> Result<Vec<HleQuestion>, LoadError> {
    let mut out = Vec::new();
    for (i, line) in jsonl.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let q: HleQuestion = serde_json::from_str(line)
            .map_err(|e| LoadError { line: i + 1, detail: e.to_string() })?;
        out.push(q);
    }
    Ok(out)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadError {
    pub line: usize,
    pub detail: String,
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "malformed question on line {}: {}", self.line, self.detail)
    }
}

impl std::error::Error for LoadError {}

/// Normalise an answer for comparison: lowercase, drop a leading "the answer
/// is" / "answer:" preface, strip punctuation to spaces, collapse whitespace.
fn normalize(s: &str) -> String {
    let lower = s.trim().to_lowercase();
    // Strip a common preface the model adds before the actual answer.
    let lower = lower
        .strip_prefix("the answer is")
        .or_else(|| lower.strip_prefix("answer:"))
        .or_else(|| lower.strip_prefix("answer is"))
        .unwrap_or(&lower);
    let mut out = String::with_capacity(lower.len());
    let mut last_space = false;
    for c in lower.chars() {
        if c.is_alphanumeric() {
            out.push(c);
            last_space = false;
        } else if !last_space {
            out.push(' ');
            last_space = true;
        }
    }
    out.trim().to_string()
}

/// Whether `given` counts as answering `expected`.
///
/// Equal after normalisation, or — for a single-token expected answer such as a
/// number or a multiple-choice letter — present as a standalone token in the
/// model's reply. Deliberately conservative: a multi-word expected answer must
/// appear as a contiguous run, so a stray matching word does not earn credit.
pub fn grade(question: &HleQuestion, given: &str) -> bool {
    let expected = normalize(&question.answer);
    let got = normalize(given);
    if expected.is_empty() {
        return false;
    }
    if got == expected {
        return true;
    }
    let exp_tokens: Vec<&str> = expected.split(' ').collect();
    let got_tokens: Vec<&str> = got.split(' ').filter(|t| !t.is_empty()).collect();
    if exp_tokens.len() == 1 {
        // A number or a choice letter: accept it as a standalone token anywhere
        // in the reply ("the correct option is c").
        got_tokens.contains(&exp_tokens[0])
    } else {
        // Multi-word: require the whole phrase as a contiguous run.
        got_tokens
            .windows(exp_tokens.len())
            .any(|w| w == exp_tokens.as_slice())
    }
}

/// The outcome of grading a whole set.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HleResult {
    pub total: u32,
    pub correct: u32,
}

impl HleResult {
    /// Fraction correct in `[0, 1]`; zero over an empty set.
    pub fn score(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.correct as f64 / self.total as f64
        }
    }
}

/// Tally graded answers into a result. `answers` is the model's reply per
/// question, positionally aligned with `questions`.
pub fn tally(questions: &[HleQuestion], answers: &[String]) -> HleResult {
    let mut correct = 0;
    for (q, a) in questions.iter().zip(answers) {
        if grade(q, a) {
            correct += 1;
        }
    }
    HleResult {
        total: questions.len().min(answers.len()) as u32,
        correct,
    }
}

/// Tally from per-item verdicts (an LLM judge's, say), positionally aligned with
/// the questions — the counterpart of [`tally`] when grading is not the built-in
/// normalised match.
pub fn tally_verdicts(verdicts: &[bool]) -> HleResult {
    HleResult {
        total: verdicts.len() as u32,
        correct: verdicts.iter().filter(|&&v| v).count() as u32,
    }
}

/// How answers were graded — printed alongside the score so a judge-graded
/// number is never mistaken for a normalised-match one (they are not comparable).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GradeMethod {
    /// Normalised string/token match — the deterministic floor ([`grade`]). Cheap
    /// and reproducible, but under-credits a correct answer phrased unusually.
    NormalizedMatch,
    /// An LLM judge decided answer-equivalence against the gold answer — what real
    /// HLE does, at one model call per item.
    LlmJudge,
}

impl std::fmt::Display for GradeMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            GradeMethod::NormalizedMatch => "normalized-match",
            GradeMethod::LlmJudge => "llm-judge",
        })
    }
}

/// The `(system, user)` prompt for an LLM judge deciding whether `given` answers
/// `question` correctly, against the gold answer.
///
/// The judge is *shown the gold answer* and asked only to check equivalence — it
/// does not solve the problem. That keeps the harness-owns-the-oracle discipline
/// intact: the solver never sees the gold answer, and the judge is not the solver
/// certifying its own success — it compares a candidate against ground truth the
/// harness holds. Real HLE grades this way; the normalised [`grade`] is the
/// deterministic floor for when no judge is available.
pub fn judge_prompt(question: &HleQuestion, given: &str) -> (String, String) {
    let system = "You are a strict grading judge. You are shown a question, the \
        correct answer, and a candidate response. Decide only whether the \
        candidate's FINAL answer means the same as the correct answer — ignore \
        phrasing, order, formatting, symbols vs words, and any working shown. Do \
        NOT solve the question yourself, and do not be swayed by confident \
        wording. Reply with exactly one word: yes or no."
        .to_string();
    let user = format!(
        "Question:\n{}\n\nCorrect answer:\n{}\n\nCandidate response:\n{}\n\n\
         Does the candidate's final answer match the correct answer? Answer yes or no.",
        question.question, question.answer, given
    );
    (system, user)
}

/// Parse a judge reply into a verdict: the last standalone `yes`/`no` token after
/// any `</think>` block (the judge's final word). `None` when the reply states
/// neither — the caller should fall back to the deterministic [`grade`] rather
/// than guess.
pub fn parse_verdict(reply: &str) -> Option<bool> {
    let body = match reply.rfind("</think>") {
        Some(i) => &reply[i + "</think>".len()..],
        None => reply,
    };
    let mut verdict = None;
    for tok in body.to_lowercase().split(|c: char| !c.is_alphanumeric()) {
        match tok {
            "yes" => verdict = Some(true),
            "no" => verdict = Some(false),
            _ => {}
        }
    }
    verdict
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q(answer: &str, ty: AnswerType) -> HleQuestion {
        HleQuestion { id: "x".into(), question: "?".into(), answer: answer.into(), answer_type: ty }
    }

    #[test]
    fn a_number_is_credited_through_a_preface() {
        let question = q("42", AnswerType::ExactMatch);
        assert!(grade(&question, "The answer is 42."));
        assert!(grade(&question, "42"));
        assert!(!grade(&question, "The answer is 43."));
    }

    #[test]
    fn a_multiple_choice_letter_is_found_in_the_reply() {
        let question = q("C", AnswerType::MultipleChoice);
        assert!(grade(&question, "The correct option is C) the third one"));
        assert!(!grade(&question, "I would pick B"));
    }

    #[test]
    fn a_multi_word_answer_needs_the_whole_phrase() {
        let question = q("Marie Curie", AnswerType::ExactMatch);
        assert!(grade(&question, "It was marie curie who discovered it."));
        // A single overlapping word must not earn credit.
        assert!(!grade(&question, "curie units are unrelated here"));
    }

    #[test]
    fn case_and_punctuation_do_not_matter() {
        let question = q("Paris", AnswerType::ExactMatch);
        assert!(grade(&question, "PARIS!"));
        assert!(grade(&question, "  paris  "));
    }

    #[test]
    fn an_empty_expected_answer_never_matches() {
        let question = q("", AnswerType::ExactMatch);
        assert!(!grade(&question, "anything"));
    }

    #[test]
    fn load_skips_blanks_and_reads_fields() {
        let jsonl = "\
{\"id\":\"1\",\"question\":\"2+2?\",\"answer\":\"4\",\"answer_type\":\"exactMatch\"}

{\"id\":\"2\",\"question\":\"pick\",\"answer\":\"B\",\"answer_type\":\"multipleChoice\"}
";
        let qs = load(jsonl).unwrap();
        assert_eq!(qs.len(), 2);
        assert_eq!(qs[0].answer, "4");
        assert_eq!(qs[1].answer_type, AnswerType::MultipleChoice);
    }

    #[test]
    fn a_malformed_line_fails_loudly() {
        match load("{\"question\":\"q\",\"answer\":\"a\"}\nnot json\n") {
            Err(e) => assert_eq!(e.line, 2),
            Ok(_) => panic!("expected a parse error"),
        }
    }

    #[test]
    fn answer_type_defaults_to_exact_match_when_absent() {
        let qs = load("{\"question\":\"q\",\"answer\":\"a\"}\n").unwrap();
        assert_eq!(qs[0].answer_type, AnswerType::ExactMatch);
    }

    #[test]
    fn tally_scores_the_fraction_correct() {
        let questions = vec![
            q("42", AnswerType::ExactMatch),
            q("Paris", AnswerType::ExactMatch),
            q("C", AnswerType::MultipleChoice),
        ];
        let answers = vec!["42".into(), "London".into(), "the answer is C".into()];
        let r = tally(&questions, &answers);
        assert_eq!(r.total, 3);
        assert_eq!(r.correct, 2);
        assert!((r.score() - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn score_is_zero_over_an_empty_set() {
        assert_eq!(HleResult { total: 0, correct: 0 }.score(), 0.0);
    }

    #[test]
    fn parse_verdict_takes_the_final_yes_no_past_thinking() {
        assert_eq!(parse_verdict("<think>gold 42, candidate 42</think>\nyes"), Some(true));
        assert_eq!(parse_verdict("No, the candidate said 43."), Some(false));
        assert_eq!(parse_verdict("YES."), Some(true));
        // The final word is the verdict, even after musing.
        assert_eq!(parse_verdict("could be yes, but on reflection no"), Some(false));
        // "yesterday" is not "yes"; neither word present -> undecided.
        assert_eq!(parse_verdict("yesterday it rained"), None);
        assert_eq!(parse_verdict("I cannot tell"), None);
    }

    #[test]
    fn judge_prompt_shows_gold_and_candidate_and_asks_only_to_match() {
        let question = q("42", AnswerType::ExactMatch);
        let (system, user) = judge_prompt(&question, "the answer is forty-two");
        assert!(system.to_lowercase().contains("yes or no"));
        assert!(system.to_lowercase().contains("do not solve"));
        assert!(user.contains("42")); // gold shown to the judge
        assert!(user.contains("forty-two")); // candidate shown to the judge
    }

    #[test]
    fn tally_verdicts_counts_trues() {
        let r = tally_verdicts(&[true, false, true, true]);
        assert_eq!(r.total, 4);
        assert_eq!(r.correct, 3);
    }
}
