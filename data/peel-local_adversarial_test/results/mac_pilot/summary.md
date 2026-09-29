# Adversarial-review audit -- summary

Inputs: fixed_input_mac.jsonl. Intervals are 95% Wilson, per cell.

## fixed-input / qwen2b-20  (n = 2)

| group | k | share | 95% CI |
|---|---:|---:|---|
| no_output | 2 | 100.0% | 34.2%–100.0% |
| same_text | 0 | 0.0% | 0.0%–65.8% |
| expanded | 0 | 0.0% | 0.0%–65.8% |
| ok | 0 | 0.0% | 0.0%–65.8% |
| changed | 0 | 0.0% | 0.0%–65.8% |

| category | k | share | 95% CI |
|---|---:|---:|---|
| fixed_no_text | 1 | 50.0% | 9.5%–90.5% |
| fixed_rejected_short | 1 | 50.0% | 9.5%–90.5% |

| accepted FIXED with word similarity ≥ | k | share of calls | 95% CI |
|---|---:|---:|---|
| 0.999 | 0 | 0.0% | 0.0%–65.8% |
| 0.99 | 0 | 0.0% | 0.0%–65.8% |
| 0.95 | 0 | 0.0% | 0.0%–65.8% |

Diagnostics: done_reason {'stop': 2}; median prompt_eval_count 4854; median eval_count 611; median call time 15 s; loaded context_length [131072].

## fixed-input / qwen9b-10  (n = 2)

| group | k | share | 95% CI |
|---|---:|---:|---|
| no_output | 0 | 0.0% | 0.0%–65.8% |
| same_text | 0 | 0.0% | 0.0%–65.8% |
| expanded | 0 | 0.0% | 0.0%–65.8% |
| ok | 1 | 50.0% | 9.5%–90.5% |
| changed | 1 | 50.0% | 9.5%–90.5% |

| category | k | share | 95% CI |
|---|---:|---:|---|
| ok | 1 | 50.0% | 9.5%–90.5% |
| fixed_changed | 1 | 50.0% | 9.5%–90.5% |

| accepted FIXED with word similarity ≥ | k | share of calls | 95% CI |
|---|---:|---:|---|
| 0.999 | 0 | 0.0% | 0.0%–65.8% |
| 0.99 | 0 | 0.0% | 0.0%–65.8% |
| 0.95 | 1 | 50.0% | 9.5%–90.5% |

Diagnostics: done_reason {'stop': 2}; median prompt_eval_count 6383; median eval_count 1794; median call time 102 s; loaded context_length [131072].

## fixed-input / qwen9b-20  (n = 2)

| group | k | share | 95% CI |
|---|---:|---:|---|
| no_output | 0 | 0.0% | 0.0%–65.8% |
| same_text | 0 | 0.0% | 0.0%–65.8% |
| expanded | 1 | 50.0% | 9.5%–90.5% |
| ok | 1 | 50.0% | 9.5%–90.5% |
| changed | 0 | 0.0% | 0.0%–65.8% |

| category | k | share | 95% CI |
|---|---:|---:|---|
| ok | 1 | 50.0% | 9.5%–90.5% |
| fixed_expanded | 1 | 50.0% | 9.5%–90.5% |

| accepted FIXED with word similarity ≥ | k | share of calls | 95% CI |
|---|---:|---:|---|
| 0.999 | 0 | 0.0% | 0.0%–65.8% |
| 0.99 | 0 | 0.0% | 0.0%–65.8% |
| 0.95 | 0 | 0.0% | 0.0%–65.8% |

Diagnostics: done_reason {'stop': 2}; median prompt_eval_count 7532; median eval_count 3739; median call time 174 s; loaded context_length [131072].
