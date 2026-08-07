# Function reference

Every function in the codebase: what it does, where it lives, and where
it's called from. For the pipeline's overall shape, start with
[README.md](README.md); this file is for finding "where is X actually
implemented" once you know roughly what you're looking for.

Interactive functions (marked **interactive**) call `input()` and log
every choice via a `DecisionLog` (see `common/decisions.py`); everything
else is a pure function with no I/O side effect beyond what's explicit in
its name (`save_*`, `print_*`).

---

## `common/paths.py`

Single source of truth for every phase's file paths, keyed by `CORPUS_NAME`.

| Function | Purpose | Called from |
|---|---|---|
| `CorpusPaths(corpus_name)` | Constructor; computes `raw_dir`, `phase1_dir`, `phase2_dir`, `decisions_dir` under `data/<corpus_name>/` | Every notebook's config cell, `run_pipeline.py` |
| `.ensure_dirs()` | Creates all of the above (+ `condensation_dir()`) if missing | Config cells, `run_pipeline.py::main` |
| `.raw_txt()` | `data/<corpus>/raw/<corpus>.txt` | phase1/phase2 notebooks, `run_pipeline.py` |
| `.top_stems()` | Phase 1's top-stems TSV path | `phase1/pipeline.py::extract_top_stems` callers |
| `.glossbert_output()` | Phase 1's accepted-definitions TSV path | `phase1/pipeline.py::save_glossbert_output` callers |
| `.phase1_state_json()` | `<corpus>-phase1_state.json` path | Phase 1's final save cell; Phase 2's load cell |
| `.phase1_html()` | Phase 1's cluster-summary HTML path | Phase 1's HTML export cell |
| `.phase2_output_json()` | `informative_sentences.json` path | Phase 2's save cell |
| `.decisions_log(phase)` | `data/<corpus>/decisions/<phase>_decisions.jsonl` path | `common/decisions.py::DecisionLog.__init__` |
| `.condensation_dir()` | `data/<corpus>/phase2/condensation/` | `.ensure_dirs()`, `.condensation_paths()` |
| `.condensation_paths(rate)` | dict of all 6 per-rate output file paths (condensed text, injection report, HTML fragment/preview, human report, plain summary) | Phase 2's generate/export cells |
| `.voyant_notebook_path(rate)` | Per-rate filled Voyant notebook HTML path | Phase 2's "BUILD VOYANT NOTEBOOK" cell |
| `resources_dir()` (module function) | `resources/` folder path (shared, cross-corpus assets) | Phase 1's Voyant-settings step, `common/voyant_notebook.py` |

## `common/decisions.py`

| Function | Purpose | Called from |
|---|---|---|
| `DecisionLog(corpus_name, phase)` | Constructor; resolves the JSONL log path via `CorpusPaths` | Every notebook's config cell, `run_pipeline.py::main` |
| `.record(step, decision_type, prompt, options=, choice=, extra=)` | Appends one JSON-Lines record with a UTC timestamp; returns the record | Every interactive `run_*` function in `phase1/pipeline.py` and `phase2/condense.py` |

## `common/ollama_client.py`

Minimal wrapper for a locally running [Ollama](https://ollama.com) instance.

| Function | Purpose | Called from |
|---|---|---|
| `is_available(host=, timeout=)` | `True` if Ollama responds at all | `condense.py::run_condensation_setup` |
| `list_models(host=, timeout=)` | Installed model names | `condense.py::run_condensation_setup` |
| `generate(model, prompt, host=, timeout=)` | Blocking, non-streaming `/api/generate` call; raises `OllamaError` on failure | `condense.py::attempt_condensation_trials` |

## `common/voyant_notebook.py`

Fills `resources/PEEL-TemplateSN.html`'s derivable placeholders with a
corpus's Phase 1 + Phase 2 results (see README's "Phase 2 condensation &
injection review" section for which placeholders and why).

| Function | Purpose | Called from |
|---|---|---|
| `_to_js_array(items)` | Python list -> JS array literal string | `build_voyant_notebook` |
| `build_cluster_results_html(phase1_state)` | Renders Phase 1 clusters as an HTML list, for the template's cluster-results cell | `build_voyant_notebook` |
| `build_optional_material_html(phase1_state, corpus_name, rates, paths)` | Short summary (rates generated, decision-log paths) for the template's "optional material" cell | `build_voyant_notebook` |
| `build_voyant_notebook(phase1_state, condensation_fragment_by_rate, corpus_name, paths)` | Reads the shared template, does the placeholder substitutions, returns the filled HTML (template file itself untouched) | Phase 2's final notebook cell, `run_pipeline.py::run_phase2` |
| `save_voyant_notebook(html, path)` | Writes the filled HTML to disk | Same callers as above |

---

## `phase0/clean_corpus.py`

Corpus cleaning: removes page numbers, running headers/footers,
footnote-call digits, broken hyphenation, front matter, and end sections.
Also runnable as a CLI script (`python clean_corpus.py raw.txt -o cleaned.txt`).

| Function | Purpose | Called from |
|---|---|---|
| `remove_isolated_page_numbers(lines)` | Drops lines that are just a number | `clean_text` |
| `remove_recurring_short_lines(lines, ...)` | Drops short lines (running headers/footers) repeated `min_repeats`+ times | `clean_text` |
| `remove_footnote_calls(text)` | Strips footnote-marker digits stuck to words (`knowledge1` -> `knowledge`) | `clean_text` |
| `fix_broken_hyphenation(text)` | Rejoins words split by a line-break hyphen | `clean_text` |
| `split_juxtaposed_words(text)` | Splits merged words when both halves are valid WordNet words | `clean_text` |
| `scan_front_matter(lines, ...)` | Finds front-matter lines (masthead, copyright, abstract, ...) by content match across the whole document | `remove_front_matter` |
| `remove_front_matter(text)` | Removes the lines `scan_front_matter` flags | `clean_text` |
| `remove_end_sections(text)` | Cuts everything from a Notes/References/Bibliography heading onward | `clean_text` |
| `normalize_whitespace(text)` | Collapses runs of spaces/blank lines | `clean_text` |
| `clean_text(text)` | Runs the full pipeline in order | `phase0.ipynb`, `run_pipeline.py::place_raw_text` |
| `main()` | CLI entrypoint (`argparse`) | `python clean_corpus.py ...` |

## `phase1/pipeline.py`

Stem extraction, GlossBERT WSD, and Sentence-BERT/HDBSCAN clustering.
`run_*` functions are **interactive**.

| Function | Purpose | Called from |
|---|---|---|
| `load_glossbert(model_id=, device=)` | Loads GlossBERT from the Hugging Face Hub (`jvomiranda/GlossBERT_Checkpoint` by default) | Phase 1's GlossBERT cell, `run_pipeline.py::run_phase1` |
| `glossbert_predict(occurrence, tokenizer, model, device, ...)` | Scores a word occurrence against its candidate WordNet senses | `run_glossbert_analysis`, `recluster_large_clusters`, `recluster_noise` |
| `extract_top_stems(doc, stemmer, top_percentile, max_stems)` | Frequency-ranked stem extraction | Phase 1's stem-extraction cell, `run_pipeline.py` |
| `map_stems_to_sentences(doc, top_stems, stemmer)` | Builds `{stem: [occurrence, ...]}` | Phase 1's GlossBERT cell, `run_pipeline.py` |
| `record_accepted_instance(accepted_definitions, stem, word, pos, definition, sentence, count)` | Insert-or-append into `accepted_definitions[stem]` | `run_glossbert_analysis`, `run_flagged_term_review` |
| `run_glossbert_analysis(top_stems, stem_occurrences, tokenizer, model, device, ...)` | Runs WSD per stem; flags sense mismatches | Phase 1's GlossBERT cell, `run_pipeline.py` |
| `resolve_flagged_choice(item, choice)` | Pure resolution of one flagged-term review choice (no `input()`) | `run_flagged_term_review` |
| `save_glossbert_output(accepted_definitions, path)` | Writes the accepted-definitions TSV | Phase 1's review cell, `run_pipeline.py` |
| `print_accepted_definitions(...)` / `print_flagged_item(...)` / `print_flagged_words(...)` | Console printers | `run_glossbert_analysis` callers, `run_flagged_term_review` |
| **`run_flagged_term_review(flagged_words, accepted_definitions, decisions)`** (interactive) | Accept-all / one-by-one / selective review of flagged sense mismatches | phase1.ipynb, `run_pipeline.py::run_phase1` |
| `build_stem_embeddings(accepted_definitions, embedder_name)` | Sentence-BERT embeddings per stem instance | Phase 1's clustering cell, `run_pipeline.py` |
| `cluster_stem_embeddings(embeddings, stem_names, min_cluster_size)` | Initial HDBSCAN clustering | Same |
| `build_stem_word_frequency_table(stem_occurrences)` | `{stem: Counter(words)}` | `rename_clusters` callers |
| `rename_clusters(cluster_dict, stem_word_frequencies)` | Names a cluster by its most frequent word(s) | Naming/reclustering cells, `run_pipeline.py` |
| `merge_named_clusters(renamed_clusters, renamed_noise_clusters)` | Merges recovered noise clusters in, disambiguating name collisions | Final-naming cell, `run_pipeline.py` |
| `print_named_clusters(...)` / `print_raw_clusters(...)` | Console printers | Various clustering cells |
| `recluster_large_clusters(clusters, stem_occurrences, embedder, tokenizer, model, device, ...)` | Re-splits oversized clusters using richer context | Reclustering cell, `run_pipeline.py` |
| `extract_noise_stems(stem_names, labels)` | Stems HDBSCAN treated as noise | Noise-rerun cell, `run_pipeline.py` |
| `recluster_noise(noise_stems, stem_occurrences, embedder, tokenizer, model, device, ...)` | Re-clusters noise stems in case some form a valid smaller group | Same |
| `tokenize_many_for_analysis(nlp, texts, stemmer, ...)` | Batch tokenization for n-gram mining | `build_sentence_token_cache` |
| `build_sentence_token_cache(nlp, accepted_definitions, stemmer, ...)` | Tokenizes every accepted-definition sentence once | N-gram cell, `run_pipeline.py` |
| `normalize_cluster_stem(stem, stemmer, use_lemmas)` / `token_matches_cluster_stem(...)` | Stem/lemma matching helpers | `extract_cluster_ngrams` |
| `build_global_ngram_statistics(sentence_cache, ...)` | Corpus-wide n-gram frequency + percentile threshold | N-gram cell, `run_pipeline.py` |
| `extract_cluster_ngrams(clusters, accepted_definitions, sentence_cache, ...)` | Representative n-grams per cluster | Same |
| `attach_ngrams(renamed_clusters, cluster_ngrams)` | Merges n-grams into each cluster's dict | Same |
| `print_cluster_for_review(cluster_name, cluster_data)` | Console printer for one cluster | `run_cluster_review` |
| `parse_index_selection(raw_input, items)` | Parses `"2,5,1"` into (kept, removed); raises `ValueError` on bad input | `run_cluster_review` |
| **`run_cluster_review(renamed_clusters, decisions)`** (interactive) | Per-cluster accept/rename/remove-stems/remove-ngrams/global-exclude flow | phase1.ipynb, `run_pipeline.py::run_phase1` |
| `read_smart_stopwords(path)` | Reads Voyant's `en_smart` stopword list | `run_voyant_settings` |
| **`run_voyant_settings(decisions, stopwords_path)`** (interactive) | Corpus-ID + smart-stopwords prompts | phase1.ipynb, `run_pipeline.py::run_phase1` |
| `build_phase1_state(corpus_id, all_original_stems, ...)` | Assembles the final `phase1_state` dict (`incList`/`excList`/`clusterDefs`/...) | Phase 1's save cell, `run_pipeline.py` |
| `save_phase1_state(state, path)` | Writes `phase1_state.json` | Same |
| `hex_to_rgb(h)` / `build_cluster_html(final_clusters, corpus_name, ...)` / `save_html(html, path)` | Tableau20-colored HTML cluster summary | Phase 1's HTML export cell, `run_pipeline.py` |

## `phase2/pipeline.py`

Informative-sentence selection (lexical density + stem/n-gram matching).

| Function | Purpose | Called from |
|---|---|---|
| `load_phase1_state(path)` | Loads `phase1_state.json` | Phase 2's load cell, `run_pipeline.py` |
| `segment_corpus(nlp, text)` | Sentence-segments the corpus | `enrich_with_informative_sentences` |
| `lexical_density(doc)` | Content-word share of one sentence | `compute_lexical_densities`, `select_representative_sentences` |
| `compute_lexical_densities(sentence_docs, percentile)` | Per-sentence density + the percentile threshold | `enrich_with_informative_sentences` |
| `stem_matches(doc, cluster_stems, stemmer)` | Which cluster stems appear in a sentence | `select_representative_sentences` |
| `build_phrase_matcher(nlp, ngrams)` | spaCy `PhraseMatcher` for a cluster's n-grams | `select_representative_sentences`, `highlight_sentence` |
| `highlight_sentence(doc, matched_stems, matched_ngrams, nlp, stemmer)` | Wraps matched spans in `****...****` markers | `select_representative_sentences` |
| `score_sentence(density, matched_stems, matched_ngrams)` | `density * (1 + stems + ngrams)` | `select_representative_sentences` |
| `select_representative_sentences(cluster, sentence_docs, densities, ...)` | Top-N scored sentences for one cluster | `enrich_with_informative_sentences` |
| `enrich_with_informative_sentences(phase1_state, text, nlp, stemmer, top_n, density_percentile)` | Orchestrates the above for every cluster | Phase 2's selection cell, `run_pipeline.py` |
| `save_informative_sentences(state, path)` | Writes `informative_sentences.json` | Phase 2's save cell, `run_pipeline.py` |

## `phase2/condense.py`

LLM condensation generation + verification (verbatim scan, F/T/R/C
classification, borderline flagging, cluster coverage). `run_*` functions
are **interactive**. See README's condensation section for the taxonomy.

| Function | Purpose | Called from |
|---|---|---|
| `gather_ordered_informative_sentences(phase1_state)` | Flattens/dedupes/orders every cluster's informative sentences | Phase 2's generate cell, `run_pipeline.py` |
| `gather_cluster_key_terms(phase1_state)` | `{cluster_name: stems+ngrams}` | Same |
| `count_words(text)` / `target_word_count(source_text, rate_pct)` | Word counting / target-word-count from a rate | `attempt_condensation_trials` and callers |
| `build_condensation_prompt(ordered_sentences, cluster_key_terms, target_words, corpus_name, source_text=)` | Builds the LLM prompt (escalated version includes `source_text`) | `attempt_condensation_trials` |
| `attempt_condensation_trials(..., model, max_trials, include_full_text, ...)` | Up to `max_trials` generation attempts against Ollama, no `input()` | `run_generate_for_rate` |
| **`run_condensation_setup(decisions, host=)`** (interactive) | Ollama availability check + rate(s)/model/max-trials prompts | phase2.ipynb, `run_pipeline.py::resolve_condensation_config` |
| **`run_generate_for_rate(rate, ..., decisions, host=)`** (interactive) | Non-escalated trials, then the escalation prompt/retry if needed | phase2.ipynb, `run_pipeline.py::run_phase2` |
| `in_source(chunk_words, source_lower)` | Whole-phrase verbatim membership test | `classify_span`, `scan_verbatim_overlap` |
| `scan_verbatim_overlap(condensed_text, source_text, ...)` | Longest-run token-alignment scan against the source | `compute_injection_stats` |
| `_content_words(text, extra_function_words=)` | Content (non-function-word) tokens of a string | `classify_span`, `flag_borderline_classifications` |
| `is_metalinguistic(sentence)` | Keyword check for "about the text's own argument" sentences | `classify_span` |
| `classify_span(sent, source_sentences, source_lower, span_id)` | F/T/R/C classification of one sentence (`sent` is a spaCy `Span`, not a string; `None` if verbatim). F is POS/dependency-based (`CONNECTIVE_POS`/`_is_connective_token`); T adds a length-only safety net for short non-metalinguistic spans | `classify_condensation` |
| `classify_condensation(condensed_text, source_text, nlp)` | Classifies every body-prose sentence, block by block | Phase 2's injection-analysis cell, `run_pipeline.py` |
| `flag_borderline_classifications(all_spans)` | Flags F/T spans denser than their own definition allows | Same |
| **`run_injection_review(all_spans, borderline_flags, rate, decisions)`** (interactive) | Per-flag keep-or-reclassify prompt | phase2.ipynb, `run_pipeline.py::run_phase2` |
| `compute_injection_stats(all_spans, condensed_text, source_text)` | `non_injected_pct` + `verbatim_overlap_pct` (two independent checks) | Phase 2's injection-analysis cell, `run_pipeline.py` |
| `compute_cluster_coverage(phase1_state, condensed_text, stemmer, target_pct=)` | Per-cluster stem/n-gram coverage vs. target, with OK/WARN/DARK status | Phase 2's coverage cell, `run_pipeline.py` |

## `phase2/condensation_report.py`

Renders `condense.py`'s results into the Spyral-paste-ready HTML fragment
and the plainer report formats.

| Function | Purpose | Called from |
|---|---|---|
| `parse_condensed_blocks(condensed_text)` | Splits condensed text into typed blocks (`h1`/`authors`/`h2`/`h3`/`defn`/`p`) | `classify_condensation` (condense.py), `build_condensation_fragment`, `build_plain_summary` |
| `_spans_in_block(block_text, all_spans)` | Which classified spans fall inside one block | `_render_blocks_with_toggles` |
| `render_spans(block_text, all_spans)` | Wraps classified spans in their `S['inj_*']` style | `_render_blocks_with_toggles` |
| `render_c_toggle(span_id, source_texts)` / `render_r_toggle(span_id, source_text)` | Collapsed `<details>` source-reveal toggles | `_render_blocks_with_toggles` |
| `_render_blocks_with_toggles(body_blocks, all_spans, source_sentences)` | Renders every body block with inline toggles after C/R spans | `build_condensation_fragment` |
| `build_meta_legend(...)` | Metadata table + F/T/R/C color legend | `build_condensation_fragment` |
| `build_coverage_table(coverage_report)` | Cluster-coverage HTML table | `build_condensation_fragment` |
| `build_condensation_fragment(condensed_text, all_spans, source_sentences, coverage_report, ...)` | Assembles the full inline-style HTML fragment | Phase 2's export cell, `run_pipeline.py`, `voyant_notebook.build_voyant_notebook` |
| `build_standalone_preview(fragment_html, corpus_name)` | Wraps the fragment in a minimal browsable `<html>` | Same callers |
| `build_human_report(all_spans, borderline_flags, coverage_report, ...)` | Plain-text verification report | Same |
| `build_plain_summary(blocks)` | Markup-free title/body text | Same |
| `save_condensation_outputs(paths_dict, fragment, preview, report, summary, injection_report_data)` | Writes 5 of the 6 per-rate output files (condensed text is saved separately) | Phase 2's export cell, `run_pipeline.py` |

---

## `run_pipeline.py` (repo root)

Runs Phase 0 (optional) -> Phase 1 -> Phase 2 -> Voyant export on one new
corpus in a single script -- see README's "Running the pipeline" for usage.

| Function | Purpose |
|---|---|
| `parse_args()` | CLI flags: `--corpus`, `--input`, `--clean`, one flag per notebook config constant, optional `--rates`/`--ollama-model`/`--max-trials` |
| `place_raw_text(args, paths)` | Copies `--input` to `paths.raw_txt()`, or runs `clean_corpus.clean_text` first if `--clean` |
| `resolve_condensation_config(args, decisions)` | Uses CLI condensation flags if all three given (logged with `"source": "cli"`); else falls back to `condense.run_condensation_setup` |
| `run_phase1(args, paths, decisions)` | Mirrors `phase1.ipynb` cell-by-cell; returns `(phase1_state, nlp, stemmer, text)` |
| `run_phase2(args, paths, decisions, phase1_state, nlp, stemmer, text)` | Mirrors `phase2.ipynb` cell-by-cell |
| `main()` | Wires `CorpusPaths`, `place_raw_text`, `run_phase1` (its own `DecisionLog(phase="phase1")`), `run_phase2` (its own `DecisionLog(phase="phase2")`) |
