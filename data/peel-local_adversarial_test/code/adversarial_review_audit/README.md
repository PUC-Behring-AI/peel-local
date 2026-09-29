# Adversarial-review audit

How often does Phase 2's adversarial reviewer return **nothing usable**, or
return **the text it was given**? This directory measures that on the paper's
three configurations, so the rate can be stated in `LIMITATIONS.md` rather
than described by example.

It changes no pipeline code and writes nothing under `data/`, so the
behaviour freeze (`CONTRIBUTING.md`) is untouched.

## Why this needs new runs

The pipeline does not log the reviewer's raw response. When a `FIXED` text
is rejected, it is discarded, and four different events all reach the
decision log as `choice: unchanged` with `fixed_text: null`. So the committed
decision logs cannot separate the outcomes. They hold one call per
configuration (n = 3 in total), which is enough to show the defect exists but
not to estimate how often it happens.

## What is measured

Each reviewer call falls into exactly one of nine categories
(`classify.py`), which roll up into five groups:

| group | categories | what the researcher sees |
|---|---|---|
| `no_output` | `call_failed`, `empty_response`, `unparseable`, `fixed_no_text`, `fixed_rejected_short` | logged `unchanged`; the reviewer's output, if there was any, is gone |
| `same_text` | `fixed_identical` | logged **`fixed`**, but the text is the one it was given (up to whitespace) |
| `expanded` | `fixed_expanded` | an accepted "fix" longer than 1.05 × max(reviewed text, target) — PEEL's own 5% length tolerance; there is no community standard for this |
| `ok` | `ok` | the reviewer found no defect |
| `changed` | `fixed_changed` | an edited text was applied |

`fixed_rejected_short` is the L2 path: the model did produce a text, and the
pipeline threw it away. It is kept as a separate category because it is a
code defect, not a model one.

"Same text" is reported two ways: exact equality (`fixed_identical`), and
word-level similarity of accepted fixes at the thresholds 0.999, 0.99 and
0.95 (`summary_similarity.csv`), which catches a full stop added or a word
swapped. For rejected fixes, `containment` shows whether the "fix" is a
piece of the original cut short.

The classifier uses the pipeline's own parser and acceptance rule. Every call
is also checked against the decision the real pipeline recorded for it, and
`analyze.py` exits non-zero on any disagreement. `test_classify.py` pins that
equivalence without Ollama.

## Designs

| | `fixed-input` | `pipeline` |
|---|---|---|
| reviewer input | the exact condensation each paper run reviewed | a fresh generation per replicate |
| isolates | the reviewer's own variability | reviewer and generation together |
| use | the rate reported in LIMITATIONS | checks that the rate carries over to new texts |
| n | 385 per configuration → 95% CI half-width ≤ 5 pp | smaller; a generalisation check |

The reviewer is called with the pipeline's settings: no `temperature`, no
`seed`, `think=False`, no `options`, timeout 600 s. Its sampling therefore
varies from call to call, which is what makes repeating it on one input
informative. In `pipeline` mode, each configuration uses its paper run's
recorded escalation choice (qwen2b escalated; the qwen9b runs did not). The
paper runs never reached the sanity-review step. When a replicate does, the
candidate closest to the target is accepted, and the replicate is tagged
`sanity_candidate_auto_accepted`.

Configurations (generator = reviewer in all three, as in the paper):

| key | run | model |
|---|---|---|
| `qwen2b-20` | `Boisseau-qwen2b-25cond-20informativesents` | `qwen3.5:2b` |
| `qwen9b-10` | `Boisseau-qwen9b-25cond-10informativesents` | `qwen3.5:9b` |
| `qwen9b-20` | `Boisseau-qwen9b-25cond-20informativesents` | `qwen3.5:9b` |

## Running

From the repository root, with Ollama running and both models pulled:

```bash
# 1. pilot: 2 calls per configuration, to measure time per call
python experiments/adversarial_review_audit/audit.py fixed-input --n 2

# 2. the main run (resumable: re-running continues where it stopped)
python experiments/adversarial_review_audit/audit.py fixed-input --n 385

# 3. generalisation check
python experiments/adversarial_review_audit/audit.py pipeline --n 30

# 4. tables
python experiments/adversarial_review_audit/analyze.py
```

Outputs go to `results/`: one JSONL per design (a line per call, raw
response included) and one `env_*.json` per session. The env file records
the Ollama version, model digests and parameters, platform and repository
commit.

## How the reported run was done

The results in `results/` came from SageMaker training jobs (`sagemaker/`).
Ollama 0.24.0 was installed at job start. Both model digests were checked
against the Mac pilot's digests, and the job aborted if they differed. The
run used `OLLAMA_CONTEXT_LENGTH=32768`, and `results/aws/summary_integrity.csv`
shows no call reached that limit. The pilot ran 2 calls per configuration on
all six allowed GPU types. The main run used 2 × A10G. Ollama 0.24.0 serves
`qwen35` one request at a time, so parallelism came from using more than one
machine, not from concurrent requests on one GPU. See `results/RESULTS.md`.

## A dependency worth knowing about

The reviewer call passes no `num_ctx`. Generation does
(`estimate_num_ctx`). The reviewer prompt is roughly 40–45k characters, and a
full `FIXED` answer repeats about 20k more. So the context the reviewer
actually gets is Ollama's default for the installed version and machine. That
default may be smaller than prompt plus answer. If it is, the rates measured
here are properties of this Ollama/hardware combination as much as of the
model. Each record keeps `prompt_eval_count`, `eval_count`, `done_reason` and
the loaded model's `context_length` (from `/api/ps`), so this can be read off
the results instead of guessed.

In the reported run, no call was truncated. The Mac pilot loaded 131,072
tokens of context, and every AWS call stayed under 32,768 with
`done_reason: stop`.
