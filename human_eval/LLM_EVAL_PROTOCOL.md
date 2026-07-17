# LLM Evaluation Protocol (LaMP-4 / LaMP-7)

## Purpose
Use an LLM as an automatic evaluator to compare DPS vs CAD outputs.
This complements human evaluation and provides richer supporting ratings.

## Inputs
Each JSONL record (from `decore/human_eval/LaMP-4_dps_vs_cad.jsonl` and
`decore/human_eval/LaMP-7_dps_vs_cad.jsonl`) includes:
- `profile`: user profile/context
- `input`: task input
- `dps_prediction`, `cad_prediction`: candidate outputs

Before evaluation, randomize which candidate is labeled A vs B and store
the mapping for later analysis.

## Prompt Template
System:
You are a careful evaluator. Follow the rubric and respond in JSON only.

User:
Task: {TASK_NAME}
Profile:
{PROFILE}

Input:
{INPUT}

Output A:
{OUTPUT_A}

Output B:
{OUTPUT_B}

Please judge:
1) Which output is better overall for this task?
2) Which output is more aligned with the profile?
3) Provide 1-5 ratings for each output on the criteria below.

Return JSON with this schema:
{
  "win": "A|B|tie",
  "alignment_win": "A|B|tie",
  "ratings": {
    "A": {"relevance": 1-5, "fluency": 1-5, "style": 1-5, "alignment": 1-5, "factuality": 1-5},
    "B": {"relevance": 1-5, "fluency": 1-5, "style": 1-5, "alignment": 1-5, "factuality": 1-5}
  },
  "notes": "brief reason"
}

## Rubric (1-5)
- Relevance / meaning: matches the input intent and preserves meaning.
- Fluency: grammatical, coherent, natural.
- Style: fits task style (headline for LaMP-4, tweet-like for LaMP-7).
- Alignment: reflects the user's profile preferences and constraints.
- Factuality / consistency: avoids hallucinations or contradictions with input.

## Task-Specific Guidance
### LaMP-4 (Headline Generation)
Prefer outputs that are concise, headline-like, and faithful to the input,
while matching the user's profile topics or style.

### LaMP-7 (Tweet Paraphrase)
Prefer outputs that preserve meaning, read like a natural tweet,
and match the user's profile style or tone.

## Decoding Settings for LLM Evaluator
Use deterministic decoding to reduce variance:
- temperature: 0
- top_p: 1.0
- max_tokens: small (just enough for JSON)

Optionally run 3 independent passes and majority vote for `win` and
`alignment_win`.

## Aggregation
Per task:
- Win rate (DPS vs CAD), ties reported separately.
- Alignment win rate (DPS vs CAD).
- Mean ratings per method for each criterion.

Report overall means and per-task breakdowns.
