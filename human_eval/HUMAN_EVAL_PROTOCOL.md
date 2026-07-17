# Human Evaluation Protocol (LaMP-4 / LaMP-7)

## Overview
We compare two decoding methods on personalized generation:
- DPS (our method)
- Context-Aware Decoding (CAD)

Annotators see an input and two candidate outputs, then judge which output is better.
We use 100 samples from LaMP-4 and 100 samples from LaMP-7.

## Materials
- LaMP-4 samples: `decore/human_eval/LaMP-4_dps_vs_cad.jsonl`
- LaMP-7 samples: `decore/human_eval/LaMP-7_dps_vs_cad.jsonl`

Each JSONL record contains:
- `input`: the task input text
- `dps_prediction`: DPS output
- `cad_prediction`: CAD output
- `profile`: user profile/context used for personalization


## Presentation Rules
- Blind the methods: randomize which output is shown as A vs B.
- Do not reveal which method produced which output.
- Randomize the order of items per annotator.
- Always show the `profile` before the input and outputs.

## Preference Alignment Guidance
Preference alignment means: the output reflects the user's stated preferences, style, or constraints
from the `profile`, while still answering the input. Judge alignment even if both outputs are fluent.
Examples of misalignment include contradicting stated dislikes, ignoring stated tone/style, or
choosing content that conflicts with the user's typical interests.

## Task-Specific Guidance
### LaMP-4 (Headline Generation)
Pick the output that is:
- Most relevant to the input article
- Concise and headline-like
- Fluent and grammatically correct
- Not misleading or hallucinated
- Most aligned with the user profile (topic focus or style)

### LaMP-7 (Tweet Paraphrase)
Pick the output that:
- Preserves the original meaning
- Sounds like a natural tweet (informal style is ok)
- Is fluent and not awkward
- Avoids adding or removing important content
- Best matches the user profile's tone or style preferences

## Annotation Decision
Primary decision (required):
- A is better
- B is better
- Tie / indistinguishable

Alignment decision (required):
- A is more aligned with the profile
- B is more aligned with the profile
- Tie / indistinguishable

## Quality Control
- Each item should be rated by 3 annotators.
- Use majority vote; if no majority, adjudicate with a 4th rater.
- Track inter-annotator agreement (e.g., Krippendorff alpha).

## Reporting
For each task:
- Win rate of DPS vs CAD (ties reported separately)
- Alignment win rate of DPS vs CAD (ties reported separately)
- Mean optional ratings by method (if collected)
- Agreement statistics

## Results Summary (Current)

Results below are computed from the annotation aggregates:
`decore/human_eval/LaMP-4_win_rates.json` and `decore/human_eval/LaMP-7_win_rates.json`.

**LaMP-4 (50 usable rows)**  
Primary decision:
- DPS: 19 (38%)
- CAD: 17 (34%)
- Tie: 14 (28%)

Alignment decision:
- DPS: 20 (40%)
- CAD: 17 (34%)
- Tie: 13 (26%)

**LaMP-7 (50 usable rows; 55 total annotations with 5 flagged as damaged)**  
Primary decision:
- DPS: 14 (28%)
- CAD: 21 (42%)
- Tie: 15 (30%)

Alignment decision:
- DPS: 14 (28%)
- CAD: 19 (38%)
- Tie: 17 (34%)
