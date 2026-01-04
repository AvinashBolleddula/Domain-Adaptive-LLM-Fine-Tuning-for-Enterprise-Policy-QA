def prompt_template(data: str, num_records: int = 6):
    return f"""
You are creating a HIGH-QUALITY instruction-tuning dataset from an internal HR Leave & Benefits policy excerpt.

CRITICAL RULES (must follow):
1) Use ONLY the information in the provided excerpt. Do NOT use outside knowledge, laws, or assumptions.
2) Prefer questions that test precise policy details that are *directly checkable* in the excerpt:
   - exact numbers, thresholds, time increments, rounding rules
   - system names / tools
   - definitions that appear in the excerpt
   - explicit do/don't rules
3) Answers must be short, direct, and policy-grounded (1 sentence preferred).
4) If (and only if) the excerpt does NOT contain the answer, write EXACTLY:
   "Not specified in the provided excerpt."
   Do NOT guess.
5) Output must be STRICT JSON only (no markdown, no commentary, no extra text).

TASK:
Generate exactly {num_records} Q/A objects total, split like this:
- 70% ANSWERABLE (KILLER): Questions whose answers are clearly present AND specific in the excerpt.
- 30% UNANSWERABLE (TRAP): Questions that sound plausible BUT are NOT answered in the excerpt.

KILLER QUESTION REQUIREMENTS (ANSWERABLE ones):
- At least 3 must have an exact number/unit in the answer (minutes/hours/days/percent/increments).
- At least 1 must have an answer that is a *single entity token* if present (e.g., CAPPS).
- At least 1 must involve a rule boundary like:
  - “0–7 minutes”, “8–22 minutes”, “15-minute increments”, etc. (if present in excerpt)
- Avoid generic HR advice. Each question should map to a specific sentence/table in the excerpt.

TRAP QUESTION REQUIREMENTS (UNANSWERABLE ones):
- Must be specific and realistic (not broad like “What is FMLA?”).
- Must ask for a detail that a real policy *might* contain but this excerpt does not (e.g., accrual rates, carryover caps, approval chain).
- The correct answer MUST be exactly: "Not specified in the provided excerpt."

Return JSON in this exact format:
{{
  "generated": [
    {{"question": "...", "answer": "..."}},
    ...
  ]
}}

EXCERPT:
\"\"\"{data}\"\"\"
""".strip()

if __name__ == "__main__":
    print(prompt_template("Avinash Bolleddula", 10))