# dataquality.py
# Production-grade quality gate for instruction-tuning Q/A pairs (policy-style).
# Goal: KEEP useful, policy-grounded pairs + strong "Not specified..." refusals,
# while dropping junk, non-questions, hallucination-y, or mismatched pairs.
#
# Usage:
#   uv run python dataquality.py
#
# Inputs:
#   data/instruction.json   -> list[{"question": "...", "answer": "..."}] (context optional)
#
# Outputs:
#   data/instructionquality.json  -> kept records
#   qualityresults.json           -> audit (kept+dropped with scores & reasons)
#   dropped.json                  -> dropped records only (for debugging)

import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from litellm import completion
from pydantic import BaseModel


# ---------------- Config ----------------
IN_PATH = "data/instruction.json"
OUT_KEPT = "data/instructionquality.json"
OUT_AUDIT = "qualityresults.json"
OUT_DROPPED = "dropped.json"

JUDGE_MODEL = "ollama_chat/qwen2.5:7b"  # change if needed

# Thresholds (tuned for your use-case)
KEEP_ACC_MIN = 6
KEEP_STYLE_MIN = 6

# Reduce judge calls by being generous with good-looking pairs
MAX_JUDGE_CALLS = None  # int or None for unlimited

# Speed / stability
JUDGE_RETRIES = 2
JUDGE_SLEEP_BASE = 0.5

# Canonical refusal string that you WANT to KEEP
CANONICAL_NOT_SPECIFIED = "Not specified in the provided excerpt."


# ---------------- Schemas ----------------
class Score(BaseModel):
    score: int  # 0-10
    explanation: str


class Rank(BaseModel):
    accuracy: Score
    style: Score


# ---------------- Helpers ----------------
_NOT_SPECIFIED_RE = re.compile(r"^\s*not specified in the provided excerpt\.\s*$", re.IGNORECASE)
ELLIPSIS_RE = re.compile(r"^\s*(\.\.\.|…)\s*$")
HAS_QMARK_RE = re.compile(r"\?\s*$")
QUESTION_START_RE = re.compile(
    r"^\s*(what|when|where|why|how|which|who|does|do|is|are|can|should|must|may|will)\b",
    re.IGNORECASE,
)

# very basic tokenization for overlap checks (no external deps)
WORD_RE = re.compile(r"[a-zA-Z0-9]+")


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def tokens(s: str) -> List[str]:
    return [t.lower() for t in WORD_RE.findall(s or "")]


def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


def looks_like_question(q: str) -> bool:
    qn = normalize_text(q)
    if len(qn) < 6:
        return False
    return bool(HAS_QMARK_RE.search(qn)) or bool(QUESTION_START_RE.search(qn))


def looks_harmful_or_sensitive(q: str, a: str) -> bool:
    # Keep this conservative—your prompt already avoids harmful content.
    s = (q + " " + a).lower()
    # Simple blocklist: you can expand if needed
    bad = ["how to make a bomb", "kill", "suicide", "self-harm", "weapon", "explosive"]
    return any(x in s for x in bad)


def is_short_entity_answer(a: str) -> bool:
    an = normalize_text(a)
    if len(an) > 30:
        return False
    # Allow CAPPS, HR codes, acronyms, yes/no, short tokens
    if an.lower() in {"yes", "no"}:
        return True
    if an.isupper() and 2 <= len(an) <= 12:
        return True
    if re.fullmatch(r"[A-Za-z]{2,}\d{2,}", an):  # e.g., HR0307
        return True
    if re.fullmatch(r"\d+(\.\d+)?", an):  # numeric-only
        return True
    if re.search(r"\b(minute|minutes|hour|hours|increment|increments|disregard)\b", an, re.I):
        return True
    return False


def heuristic_grade(pair: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Return (result, decision_reason).
    If result is not None, it is a Rank-like dict and we skip LLM judging.
    If result is None, caller may invoke LLM judge.
    """
    q = normalize_text(pair.get("question", ""))
    a = normalize_text(pair.get("answer", ""))

    # Basic empties
    if not q or not a or ELLIPSIS_RE.match(q) or ELLIPSIS_RE.match(a):
        return ({
            "accuracy": {"score": 1, "explanation": "Blank/ellipsis content."},
            "style": {"score": 1, "explanation": "Blank/ellipsis content."},
        }, "blank_or_ellipsis")

    # Harmful -> drop hard
    if looks_harmful_or_sensitive(q, a):
        return ({
            "accuracy": {"score": 1, "explanation": "Potentially harmful/sensitive content."},
            "style": {"score": 1, "explanation": "Potentially harmful/sensitive content."},
        }, "harmful")

    # Canonical refusal string is gold for your project
    if _NOT_SPECIFIED_RE.match(a):
        return ({
            "accuracy": {"score": 10, "explanation": "Correct refusal: explicitly marked as not in excerpt."},
            "style": {"score": 10, "explanation": "Honest and non-hallucinated."},
        }, "canonical_refusal_keep")

    # Question sanity
    if not looks_like_question(q):
        return ({
            "accuracy": {"score": 0, "explanation": "Not a real question."},
            "style": {"score": 7, "explanation": "Not harmful, but unusable as instruction."},
        }, "not_a_question")

    # Strong keep: short entity answers for system/increment/yes-no/numeric policies
    if is_short_entity_answer(a):
        # Avoid obvious mismatch: if answer tokens share almost nothing with question tokens,
        # then it might be a stray acronym unrelated to the asked topic.
        qt, at = tokens(q), tokens(a)
        if len(at) <= 3:
            # For very short answers, require minimal topical link OR allow common policy entities
            allow_entities = {"capps", "hr", "fmla", "txhhs", "hhs"}
            if (set(at) & set(qt)) or (set(at) & allow_entities):
                return ({
                    "accuracy": {"score": 9, "explanation": "Short structured/entity answer acceptable for this QA style."},
                    "style": {"score": 10, "explanation": "Clear and safe."},
                }, "short_entity_keep")
        else:
            # short phrase answer
            if jaccard(qt, at) >= 0.05:
                return ({
                    "accuracy": {"score": 9, "explanation": "Short structured answer likely correct for a policy detail question."},
                    "style": {"score": 10, "explanation": "Clear and safe."},
                }, "short_structured_keep")

    # Quick mismatch check: if answer is long but shares near-zero topical overlap with question
    qt, at = tokens(q), tokens(a)
    if len(a) > 80 and jaccard(qt, at) < 0.02:
        return ({
            "accuracy": {"score": 2, "explanation": "Answer appears unrelated to the question (very low topical overlap)."},
            "style": {"score": 8, "explanation": "Not harmful, but likely mismatched."},
        }, "low_overlap_drop")

    # Otherwise: ambiguous → judge with LLM
    return (None, "needs_judge")


def judge_record(model: str, question: str, answer: str, retries: int = 2) -> Dict[str, Any]:
    """
    LLM judge that does NOT use outside knowledge.
    It should be generous to concise policy answers and canonical refusals.
    """
    record_json = json.dumps({"question": question, "answer": answer}, ensure_ascii=False)

    prompt = f"""
You are grading an instruction-tuning (question, answer) pair that was generated from a specific HR Leave & Benefits policy excerpt.

You MUST follow these grading rules:
- Do NOT use outside/world knowledge (no assumptions about typical HR policies, laws, etc.).
- Evaluate ONLY internal consistency: does the answer directly address the question?
- SHORT answers are acceptable if they directly answer (e.g., "CAPPS", "15-minute increments", "Yes/No", "disregard", "8 minutes").
- The exact answer "{CANONICAL_NOT_SPECIFIED}" is FULLY CORRECT when the excerpt doesn't contain that info.
- Penalize when:
  * the question is not a real question,
  * the answer is blank/meaningless,
  * the answer is unrelated to the question,
  * the answer claims specific facts in a way that clearly doesn't match the question,
  * the content is harmful/dishonest/unhelpful.

Scoring guidance:
- accuracy.score 9-10: directly answers; concise is OK.
- accuracy.score 6-8: mostly answers; slight phrasing mismatch but still useful for training.
- accuracy.score 0-3: not a question, blank, or unrelated.
- style.score 9-10: safe, honest, helpful.
- style.score <=3: harmful, deceptive, unsafe, or clearly unhelpful.

Return STRICT JSON matching the provided schema.

Record JSON:
{record_json}
""".strip()

    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                options={"temperature": 0.0, "num_predict": 350},
                format=Rank.model_json_schema(),
            )
            content = resp["choices"][0]["message"]["content"]
            return json.loads(content)
        except Exception as e:
            last_err = e
            time.sleep(JUDGE_SLEEP_BASE * (attempt + 1))

    raise RuntimeError(f"Judge failed after retries. Last error: {last_err}")


def clamp_scores(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Safety: normalize score types and bounds.
    """
    for k in ("accuracy", "style"):
        score = int(result[k]["score"])
        score = max(0, min(10, score))
        result[k]["score"] = score
        result[k]["explanation"] = normalize_text(result[k].get("explanation", ""))
    return result


# ---------------- Main ----------------
def main() -> None:
    with open(IN_PATH, "r") as f:
        data: List[Dict[str, Any]] = json.load(f)

    total = len(data)
    kept: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []
    audit: List[Dict[str, Any]] = []

    judge_calls = 0
    print(f"[dataquality] grading {total} records | judge={JUDGE_MODEL}")

    for idx, pair in enumerate(data, start=1):
        q = normalize_text(pair.get("question", ""))
        a = normalize_text(pair.get("answer", ""))

        # --- Heuristic first ---
        heuristic_result, reason = heuristic_grade({"question": q, "answer": a})

        if heuristic_result is not None:
            result = clamp_scores(heuristic_result)
            used_judge = False
        else:
            # --- Judge as LAST resort ---
            if MAX_JUDGE_CALLS is not None and judge_calls >= MAX_JUDGE_CALLS:
                # If judge budget exhausted, be conservative but not destructive:
                # keep if it looks like a question and has some topical overlap, else drop.
                qt, at = tokens(q), tokens(a)
                overlap = jaccard(qt, at)
                if looks_like_question(q) and (overlap >= 0.03 or is_short_entity_answer(a)):
                    result = {
                        "accuracy": {"score": 7, "explanation": "Judge budget exhausted; kept based on heuristic plausibility."},
                        "style": {"score": 9, "explanation": "Appears safe and instruction-like."},
                    }
                    reason = "budget_keep"
                else:
                    result = {
                        "accuracy": {"score": 3, "explanation": "Judge budget exhausted; low plausibility (low overlap / not a question)."},
                        "style": {"score": 8, "explanation": "Not harmful, but likely unusable."},
                    }
                    reason = "budget_drop"
                result = clamp_scores(result)
                used_judge = False
            else:
                result = clamp_scores(judge_record(JUDGE_MODEL, q, a, retries=JUDGE_RETRIES))
                used_judge = True
                judge_calls += 1

        acc = int(result["accuracy"]["score"])
        sty = int(result["style"]["score"])

        record_audit = {
            "question": q,
            "answer": a,
            "quality": result,
            "meta": {
                "decision": "kept" if (acc >= KEEP_ACC_MIN and sty >= KEEP_STYLE_MIN) else "dropped",
                "reason": reason,
                "used_judge": used_judge,
            },
        }

        audit.append(record_audit)

        if acc >= KEEP_ACC_MIN and sty >= KEEP_STYLE_MIN:
            kept.append({"question": q, "answer": a})
        else:
            dropped.append({"question": q, "answer": a, "quality": result, "reason": reason})

        if idx % 10 == 0 or idx == total:
            print(
                f"[dataquality] {idx}/{total} | kept={len(kept)} dropped={len(dropped)} | judge_calls={judge_calls}"
            )

    with open(OUT_KEPT, "w") as f:
        json.dump(kept, f, indent=2, ensure_ascii=False)

    with open(OUT_AUDIT, "w") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)

    with open(OUT_DROPPED, "w") as f:
        json.dump(dropped, f, indent=2, ensure_ascii=False)

    print(f"[dataquality] DONE")
    print(f"  kept:   {len(kept)}/{total} -> {OUT_KEPT}")
    print(f"  audit:  {OUT_AUDIT}")
    print(f"  dropped:{OUT_DROPPED}")
    print(f"  judge calls: {judge_calls}")


if __name__ == "__main__":
    main()