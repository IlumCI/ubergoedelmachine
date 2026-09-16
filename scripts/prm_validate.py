"""Check a process-reward dataset before anything trains on it.

A mislabelled step is worse than a missing one: it teaches the reward model that
a wrong calculation is fine, and every downstream use inherits that. These are
the invariants the exporter is supposed to guarantee, verified independently
rather than trusted.

    python scripts/prm_validate.py <prm.jsonl>
"""
import json
import sys
from collections import Counter


def main(path: str) -> int:
    rows = []
    for i, line in enumerate(open(path, encoding="utf-8"), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append((i, json.loads(line)))
        except json.JSONDecodeError as e:
            print(f"line {i}: malformed JSON: {e}")
            return 1
    if not rows:
        print(f"no rows in {path}")
        return 1

    errs = []
    step_labels = Counter()
    by_source = Counter()
    by_family = Counter()
    kinds = Counter()

    for lineno, r in rows:
        def bad(msg):
            errs.append(f"line {lineno} ({r.get('id', '?')}): {msg}")

        for field in ("id", "family", "question", "answer", "steps", "labels", "source"):
            if field not in r:
                bad(f"missing field {field}")
        if errs and errs[-1].startswith(f"line {lineno}"):
            continue

        steps, labels = r["steps"], r["labels"]
        if len(steps) != len(labels):
            bad(f"{len(steps)} steps but {len(labels)} labels")
            continue
        if not steps:
            bad("no steps")
            continue
        if any(l not in (0, 1) for l in labels):
            bad(f"labels must be 0/1, got {sorted(set(labels))}")

        by_source[r["source"]] += 1
        by_family[r["family"]] += 1
        step_labels.update(labels)

        if r["source"] == "generator":
            # A positive is the generator's own path: every step correct, and
            # the trace must actually reach the stated answer.
            if any(l != 1 for l in labels):
                bad("positive example carries a 0 label")
            if r["answer"] not in steps[-1]:
                bad(f"positive does not end on its answer {r['answer']!r}: {steps[-1]!r}")
            if r.get("correct_step") is not None:
                bad("positive should not carry correct_step - there is nothing to contrast")
        elif r["source"] == "corruption":
            # Exactly one error, and it must be the last step - anything after a
            # false premise is neither right nor wrong.
            zeros = [i for i, l in enumerate(labels) if l == 0]
            if len(zeros) != 1:
                bad(f"expected exactly one 0 label, got {len(zeros)}")
            elif zeros[0] != len(labels) - 1:
                bad(f"0 label at {zeros[0]}, not the final step ({len(labels) - 1})")
            note = r.get("note", "")
            if ": " in note and " -> " in note:
                kinds[note.split(": ")[1].split(" ")[0]] += 1
                # The substituted value must genuinely differ from the truth.
                lhs = note.split(" -> ")
                truth = lhs[0].rsplit(" ", 1)[-1] if len(lhs) > 1 else ""
                wrong = lhs[-1]
                if truth and truth == wrong:
                    bad(f"corruption did not change anything: {note}")
                if wrong and wrong not in steps[-1]:
                    bad(f"corrupted value {wrong!r} absent from the broken step")
            else:
                bad(f"corruption lacks a readable note: {note!r}")
            # The matched-pair reading: same prefix, one right continuation and
            # one wrong. If they are equal the pair teaches nothing.
            cs = r.get("correct_step")
            if not cs:
                bad("negative is missing correct_step, so it cannot be read as a pair")
            elif cs == steps[-1]:
                bad("correct_step is identical to the corrupted step")
        else:
            bad(f"unknown source {r['source']!r}")

    print(f"{len(rows)} examples from {path}")
    print(f"  by source : {dict(by_source)}")
    print(f"  families  : {len(by_family)}")
    print(f"  step labels: {step_labels[1]} correct / {step_labels[0]} incorrect "
          f"({step_labels[0] / max(sum(step_labels.values()), 1):.1%} negative)")
    if kinds:
        print(f"  corruptions: {dict(kinds)}")
    pairs = sum(1 for _, r in rows if r["source"] == "corruption" and r.get("correct_step"))
    print(f"  matched pairs: {pairs} (balanced view; the step-label view is not)")

    # Every family should contribute negatives, or the reward model will be
    # blind on whichever one does not.
    neg_fams = {r["family"] for _, r in rows if r["source"] == "corruption"}
    missing = set(by_family) - neg_fams
    if missing:
        print(f"  WARNING: no negatives for {sorted(missing)} - a PRM will not "
              "learn to catch errors there")

    # Thin coverage is quieter than none and just as real: a family with few
    # negatives produces a reward model that is confidently useless on it.
    neg_by_fam = Counter(r["family"] for _, r in rows if r["source"] == "corruption")
    if neg_by_fam:
        most = max(neg_by_fam.values())
        thin = {f: n for f, n in neg_by_fam.items() if n * 3 < most}
        if thin:
            print(f"  WARNING: sparse negatives (< 1/3 of the best-covered family): {thin}")
            print("    usually means short traces, or step values that are not quoted in")
            print("    their own text and so cannot be corrupted safely")

    if errs:
        print(f"\n{len(errs)} problem(s):")
        for e in errs[:20]:
            print("  " + e)
        if len(errs) > 20:
            print(f"  ... and {len(errs) - 20} more")
        return 1
    print("\nall invariants hold")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1]))
