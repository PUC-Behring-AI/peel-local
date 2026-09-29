# Adversarial-reviewer audit (Phase 2) — PEEL-Local

*(English translation of [`LEIA-ME.md`](LEIA-ME.md); the original Portuguese is the file of record.)*

Start with **[`RESULTS.en.md`](RESULTS.en.md)**, which summarizes all the results, with protocol, tables, and the measurement's limitations.

| Path | Contents |
|---|---|
| `RESULTS.en.md` (`RESULTS.md` in Portuguese) | results summary (every number comes from the tables below) |
| `results/aws/summary_*.csv` | per-configuration tables: groups, categories, similarity, length, integrity, machine |
| `results/aws/fixed_input_aws.jsonl` | all 1,155 reviewer calls (385 per configuration), with the model's raw response |
| `results/aws/env/` | each job's environment: Ollama version, model digests, GPU |
| `results/mac_pilot/` | the Mac pilot (2 calls per configuration), for comparison only |
| `code/adversarial_review_audit/` | the audit code (`audit.py`, `classify.py`, `analyze.py`, tests, README) and SageMaker jobs |
| `code/build_summary.py` (`code/build_summary_en.py` for English) | generates `RESULTS.md`/`RESULTS.en.md` from the tables |

**Source-text rights.** The `.jsonl` contains the reviewed condensations and the model's responses, which are derived from Boisseau (2026), under CC BY-NC-ND 4.0. Redistribution permission is still pending: see `PROVENANCE_source_Boisseau.md`.
