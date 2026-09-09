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
| `CorpusPaths(corpus_name)` | Constructor; computes `raw_dir`, `phase1_dir`, `phase2_dir`, `phase3_dir`, `decisions_dir` under `data/<corpus_name>/` | Every notebook's config cell, `run_pipeline.py` |
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
| `.standalone_report_path(rate)` | Per-rate standalone report HTML path | Phase 2's "BUILD STANDALONE REPORT" cell |
| `.distant_reading_report_path()` | Phase 3's single output HTML path | `phase3/report.py::save_distant_reading_report` callers |

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
| `generate(model, prompt, host=, timeout=)` | Blocking, non-streaming `/api/generate` call; raises `OllamaError` on failure | `condense.py::attempt_condensation_trials`, `condense.py::run_adversarial_review`, `condense.py::extract_source_metadata` |

## `common/standalone_report.py`

Builds a self-contained HTML report of a corpus's Phase 1 + Phase 2
results -- condensation, cluster results, and a run summary -- as its
own cleanly styled page.

| Function | Purpose | Called from |
|---|---|---|
| `build_cluster_results_html(phase1_state)` | Renders Phase 1 clusters as an HTML list | `build_standalone_report` |
| `build_optional_material_html(phase1_state, corpus_name, rates, paths)` | Short summary (rates generated, decision-log paths) | `build_standalone_report` |
| `build_standalone_report(phase1_state, fragment, corpus_name, rate, paths)` | Combines the condensation fragment, cluster results, and run summary into one complete HTML page | Phase 2's final notebook cell, `run_pipeline.py::process_and_save_rate`, `webapp/pipeline_session.py::_process_and_save_rate` |
| `save_standalone_report(html, path)` | Writes the report HTML to disk | Same callers as above |

## `common/parameter_sweep.py`

Previews several candidate Phase 1 `frequency_percentile` values before
committing to one -- cross-phase orchestration (calls into both
`phase1/pipeline.py` and `phase2/pipeline.py`), so it lives here rather
than inside either phase's own module. The Phase 2 functions it reuses
have no dependency on real clusters existing: a single synthetic
pseudo-cluster wrapping a candidate's `top_stems` is enough for
`select_representative_sentences` to score correctly.

| Function | Purpose | Called from |
|---|---|---|
| `compute_sweep_report(doc, text, nlp, stemmer, candidates=, density_percentiles=, word_frequency_percentile=0.0)` | Pure, no `input()`: for each candidate `frequency_percentile`, runs `phase1_pipeline.extract_top_stems` (with an effectively-unbounded `max_stems` -- deliberately not the researcher's configured cap, so every candidate reports its true, uncapped stem count instead of all converging on the same ceiling)/`map_stems_to_sentences`, then (if `word_frequency_percentile > 0`) `phase1_pipeline.filter_stem_occurrences_by_word_frequency` -- so the reported distinct-word/sentence-coverage counts already reflect that filter, not the unfiltered corpus -- and `phase2_pipeline.select_representative_sentences` (via a pseudo-cluster) at each density-percentile band; returns one report row per candidate | `run_parameter_sweep_setup`, `webapp/pipeline_session.py::_step_run_parameter_sweep` |
| `format_sweep_report(rows)` | Plain-text comparison table, including the `n_distinct_words` column | `run_parameter_sweep_setup` |
| **`run_parameter_sweep_setup(doc, text, nlp, stemmer, max_stems, decisions, candidates=, density_percentiles=, word_frequency_percentile=0.0)`** (interactive) | Prints the table, prompts for a candidate number or a custom `frequency_percentile`/`max_stems`; `word_frequency_percentile` is passed straight through to `compute_sweep_report`, not swept as its own axis | phase1.ipynb, `run_pipeline.py::run_phase1` |

---

## `common/file_convert.py`

Converts a corpus input file to plain text before it's placed at
`data/<corpus>/raw/<corpus>.txt`, regardless of source format -- the one
implementation of "what counts as a supported input format" shared by
the web upload handler and `run_pipeline.py`'s `--input`.

| Function | Purpose | Called from |
|---|---|---|
| `convert_to_text(raw_bytes, filename)` | Dispatches on `filename`'s extension (`.txt`, `.md`/`.markdown`, `.pdf`); raises `ValueError` for anything else or an unextractable PDF | `webapp/pipeline_session.py::PipelineSession.start`, `run_pipeline.py::place_raw_text` |
| `_pdf_to_text(raw_bytes)` | Extracts text page by page via `pypdf` | `convert_to_text` |
| `_markdown_to_text(text)` | Regex-based markdown-to-prose pass (headings, emphasis, links/images, code fences, lists, tables, HTML tags stripped -- text kept) | `convert_to_text` |
| `_normalize_newlines(text)` | Collapses `\r\n`/`\r` to `\n` -- reading raw bytes (required for PDF support) skips Python's usual universal-newline translation | `convert_to_text` |

---

## `common/resume.py`

Lets a researcher resume an existing corpus at Phase 2 or Phase 3 instead
of always starting a brand-new corpus at Phase 1 -- shared by the CLI
(`--start-phase`/`--list-corpora`) and the webapp (the "resume an
existing corpus" setup toggle) so both use identical corpus-discovery and
prerequisite-checking logic, never two independently-drifting copies.

| Function | Purpose | Called from |
|---|---|---|
| `list_corpora()` | Every corpus directory under `data/`, sorted; `[]` if `data/` doesn't exist yet | `corpus_phase_summary`, `run_pipeline.py::_print_corpus_list`, `webapp/app.py::list_corpora` |
| `available_condensation_rates(paths)` | Rates with an existing condensed-text file, discovered directly from disk (not from any config) | `check_prerequisites`, `corpus_phase_summary` |
| `check_prerequisites(paths, start_phase)` | Returns `(ok, missing, available_rates)`. `start_phase=1` has no prerequisites (a new corpus creates its own files); `>=2` needs the saved raw text + Phase 1 state; `>=3` additionally needs at least one already-generated condensed rate. `missing` is a human-readable list for a clear abort message; `available_rates` is only populated for `start_phase==3` | `run_pipeline.py::main`, `webapp/pipeline_session.py::PipelineSession.resume`, `corpus_phase_summary` |
| `corpus_phase_summary(corpus_name)` | `{corpus_name, available_start_phases, rates}` -- which of `[1,2,3]` a corpus currently satisfies prerequisites for, and its existing condensed rates; feeds both `--list-corpora` and `GET /api/corpora` | `run_pipeline.py::_print_corpus_list`, `webapp/app.py::list_corpora` |
| `find_last_decision_choice(corpus_name, phase, decision_type)` | Scans an existing phase decision log for the most recent recorded choice of a given `decision_type` -- used when resuming to recover config (e.g. which spaCy `lang_model` Phase 1 actually used, via `phase1_pipeline.log_phase1_config`'s `lang_model_choice`) that isn't persisted anywhere else. Returns `None` if absent (e.g. a run predating that logging), so callers fall back to a default | `run_pipeline.py::_load_resume_state`, `webapp/pipeline_session.py::PipelineSession._resolve_lang_model` |

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
| `_glossbert_words_budget(gloss, tokenizer, n_marks=1, ..., max_length=128)` | How many words of context (each side, per marked occurrence) fit in a given `max_length` for a specific gloss and how many quote-marked occurrences share the budget -- `glossbert_predict` uses the default 128; `glossbert_predict_merged` passes 512 | `glossbert_predict`, `glossbert_predict_merged` |
| `_window_around_match(sentence, match_start, match_end, max_words_each_side)` | Word-based window around a match, clamped at sentence boundaries (`...` marks a real cut); a no-op when the sentence already fits. Also strips any literal `"` from the window text (replaced with a space) before the caller wraps the target word in its own `"..."` pair -- GlossBERT's only disambiguation-target signal, per its paper, so a pre-existing straight-quote pair in the context would otherwise be indistinguishable from the real marker; verified as a no-op on real Boisseau data (0 straight quotes present -- it uses typographic curly quotes, which tokenize to different ids than the marking character) | `glossbert_predict`, `glossbert_predict_merged` |
| `glossbert_predict(occurrence, tokenizer, model, device, ...)` | Scores a word occurrence against its candidate WordNet senses; marks the target word via a `\b`-anchored, case-insensitive regex search (robust when `word` is lowercase but `sentence` retains original casing, e.g. an all-caps heading; the `\b` anchoring itself fixes a real bug where an unanchored search matched a short word's letters INSIDE an unrelated earlier word, e.g. "ai" inside "main" -- verified on real Boisseau data, 6 occurrences affected), **inside a word-window sized from the longest candidate gloss** rather than the whole sentence -- otherwise HF's pair-truncation trims the sentence's tail regardless of where the target word sits, silently dropping it (and its marking) whenever it occurs late in a long sentence; verified on real data (see `docs/ai_prompts_catalog.md`); tokenized at `max_length=128` | `run_glossbert_analysis`, `recluster_large_clusters`, `recluster_noise` |
| `glossbert_predict_merged(word, occurrences, tokenizer, model, device, ...)` | Same scoring, but against one paragraph merging every one of a word's sampled occurrences (each independently windowed and marked first with the same `\b`-anchored regex fix as `glossbert_predict`, sharing one `max_length=512` budget split N ways) instead of a single occurrence -- 512 (`_GLOSSBERT_MERGED_MAX_LENGTH`) matches GlossBERT's actual BERT-base position-embedding ceiling, giving a merged paragraph much more room than the single-occurrence path's 128; used when `merge_duplicate_word_occurrences=True`, so a word yields at most one prediction instead of one per occurrence | `run_glossbert_analysis` (merge branch) |
| `extract_top_stems(doc, stemmer, frequency_percentile, max_stems)` | Frequency-ranked stem extraction: `final_n = min(ceil(vocab_size * (1 - frequency_percentile)), max_stems)`. `frequency_percentile` is a statistical percentile CUTOFF on the frequency ranking -- higher = fewer, more frequent stems (e.g. `0.75` keeps only the top 25% most-frequent). Uses rank-based selection rather than a true value-based percentile because real (Zipfian) text has huge tie-pileups at low frequencies -- verified on Boisseau, where 43.6% of stems occur exactly once, so a value-based 25th percentile (=1) would select 100% of the vocabulary instead of a predictable fraction | Phase 1's stem-extraction cell, `run_pipeline.py`, `common/parameter_sweep.py::compute_sweep_report` |
| `describe_stem_cap(top_stems, doc, stemmer, frequency_percentile, max_stems)` | Returns a one-line note when `max_stems` (not `frequency_percentile`) determined the final count -- `None` if the cap didn't bind. Printed right after "Selected N top stems" in every interface | `run_pipeline.py::run_phase1`, `phase1.ipynb`, `webapp/pipeline_session.py::_step_start_to_flagged_review` |
| `map_stems_to_sentences(doc, top_stems, stemmer)` | Builds `{stem: [occurrence, ...]}` | Phase 1's GlossBERT cell, `run_pipeline.py`, `common/parameter_sweep.py::compute_sweep_report` |
| `filter_stem_occurrences_by_word_frequency(stem_occurrences, word_frequency_percentile, min_words_per_stem=1)` | Trims each stem's distinct DERIVED WORDS (not occurrences) to those at/above a percentile cutoff of that stem's OWN word-frequency ranking -- same rank-based cutoff semantics as `frequency_percentile`, applied one level down (per stem, not pooled globally), so a low-frequency stem can't be emptied out by a high-frequency stem's common forms. Always keeps >= `min_words_per_stem` words per stem. `word_frequency_percentile<=0` (default everywhere) is a no-op | Phase 1's GlossBERT cell, `run_pipeline.py::run_phase1`, `webapp/pipeline_session.py::_step_start_to_flagged_review` |
| `PHASE1_CONFIG_PROMPTS` / `log_phase1_config(decisions, config, source)` | One shared prompt string per Phase 1 config parameter (12 total: `frequency_percentile`, `max_stems`, `word_frequency_percentile`, `max_sentences_per_stem`, `max_synsets`, `merge_duplicate_word_occurrences`, `max_cluster_size`, `min_clusters`, `min_cluster_len`, `glossbert_model`, `lang_model`, `sentence_embedder`), so all three interfaces log identical wording. `log_phase1_config` records one `step="phase1_setup"`/`decision_type="{param}_choice"` decision per key in `config` -- so the full resolved configuration a run actually used (not just the interactive review choices) is reconstructable from `phase1_decisions.jsonl` alone | `run_pipeline.py::run_phase1`, `phase1.ipynb`, `webapp/pipeline_session.py::_step_start_to_flagged_review` |
| `record_accepted_instance(accepted_definitions, stem, word, pos, definition, sentence, count)` | Insert-or-append into `accepted_definitions[stem]` | `run_glossbert_analysis`, `run_flagged_term_review` |
| `purge_word_from_occurrences(stem_occurrences, stem, word)` | Removes every occurrence of `word` from `stem_occurrences[stem]` in place | `run_flagged_term_review`'s/`apply_flagged_term_review`'s "delete" choice |
| `run_glossbert_analysis(top_stems, stem_occurrences, tokenizer, model, device, ..., merge_duplicate_word_occurrences=False)` | Runs WSD per stem (words lowercased, `PROPN` mapped to `wn.NOUN`); flags sense mismatches, deduped per stem by `(word, default_sense, predicted_sense)` so e.g. the same word from an all-caps heading and from normal prose only produces one review item; `top_candidates` respects `max_synsets`. When `merge_duplicate_word_occurrences=True`, occurrences are grouped by exact word text first and scored once per word via `glossbert_predict_merged` instead of once per occurrence, closing the gap where the same word landing on two different predicted senses still produced two review items | Phase 1's GlossBERT cell, `run_pipeline.py` |
| `describe_wordnet_excluded_stems(stem_occurrences, max_sentences_per_stem=5, merge_duplicate_word_occurrences=False)` | Replicates `run_glossbert_analysis`'s own `if not synsets: continue` skip (same occurrence cap, same merged-vs-non-merged POS lookup -- the merged branch keys a word's WordNet POS off only its FIRST sampled occurrence, so it must be called with the SAME `max_sentences_per_stem`/`merge_duplicate_word_occurrences` the analysis run actually used, not just any occurrence containing an eligible POS somewhere) to report every selected stem left with zero WordNet-eligible words -- these produce no flagged item and no accepted instance, so they silently vanish between "Selected N top stems" and the final embedded count with no other trace. Returns `None` if every stem has at least one eligible word. Printed right after "Done. N flagged term(s) need review." in every interface. Verified against a real run (`merge_duplicate_word_occurrences=True`): correctly flags `defer`/`et`/`non`/`veritist` (each stem's only WordNet-checked occurrence has 0 synsets at that POS) and correctly leaves out `ross` (has real synsets -- it was a genuine flagged-then-deleted case during review, not a silent one) | `run_pipeline.py::run_phase1`, `phase1.ipynb`, `webapp/pipeline_session.py::_step_start_to_flagged_review` |
| `resolve_flagged_choice(item, choice)` | Pure resolution of one flagged-term review choice (no `input()`); candidate slots and the manual-entry slot are keyed off `len(item["top_candidates"])`, not a fixed count; also resolves `"delete"` (remove the word from all following steps) | `run_flagged_term_review` |
| `save_glossbert_output(accepted_definitions, path)` | Writes the accepted-definitions TSV | Phase 1's review cell, `run_pipeline.py` |
| `print_accepted_definitions(...)` / `print_flagged_item(...)` / `print_flagged_words(...)` | Console printers | `run_glossbert_analysis` callers, `run_flagged_term_review` |
| **`run_flagged_term_review(flagged_words, accepted_definitions, decisions, stem_occurrences=None)`** (interactive) | Accept-all / one-by-one / selective review of flagged sense mismatches, including a per-item "delete" choice that skips acceptance and (if `stem_occurrences` is given) also purges the word via `purge_word_from_occurrences`, so it can't resurface in cluster naming or a later resplit | phase1.ipynb, `run_pipeline.py::run_phase1` |
| `_fit_contexts_to_budget(tokenizer, max_seq_length, fixed_parts, contexts, min_words_per_context=3)` | Caps a list of context sentences so the whole embedding text fits `max_seq_length`, computed from THIS text's own real tokenization at call time (not a fixed word count from any one corpus) -- measures every other template section's actual token cost, gives contexts whatever budget remains, then caps each sentence using its own real tokens-per-word ratio; adapts automatically across sources with different vocabularies/languages. Its internal length-measuring calls use `truncation=True, max_length=max_seq_length` -- purely defensive (the measured value is only ever compared against a smaller per-sentence share, so capping it changes no decision), but silences transformers' "Token indices sequence length..." warning that a genuinely long raw sentence (verified on real Boisseau data: 295+ tokens for one real sentence) would otherwise print on every such measurement, despite the final embedded text never actually exceeding `max_seq_length` | `build_stem_embeddings`, `_split_oversized_clusters_once`, `recluster_noise` |
| `build_stem_embeddings(accepted_definitions, embedder_name)` | Sentence-BERT embeddings per stem, built from accepted instances' observed words/adaptively-capped contexts/definitions | Phase 1's clustering cell, `run_pipeline.py` |
| `cluster_stem_embeddings(embeddings, stem_names, min_cluster_size)` | Initial HDBSCAN clustering | Same |
| `build_stem_word_frequency_table(stem_occurrences)` | `{stem: Counter(words)}` | `rename_clusters` callers |
| `rename_clusters(cluster_dict, stem_word_frequencies)` | Names a cluster by its most frequent word(s); collisions (two different clusters sharing the same dominant word -- common for a corpus-wide high-frequency term) are disambiguated with the same numeric-suffix strategy `merge_named_clusters` uses (`"trust"`, `"trust (2)"`, ...), not silently overwritten. Verified on real Boisseau data: before this fix, 6 of 20 raw noise clusters collided on `trust`/`agent`/`systems`/`expertise` and were dropped entirely (a plain dict assignment let the last-processed cluster silently erase the earlier same-named one's stems); after the fix all 20 survive under disambiguated names | Naming/reclustering cells, `run_pipeline.py` |
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
| **`run_cluster_review(renamed_clusters, decisions)`** (interactive) | Per-cluster accept/rename/remove-stems/remove-ngrams flow | phase1.ipynb, `run_pipeline.py::run_phase1` |
| `build_phase1_state(final_clusters, excluded_cluster_ngrams)` | Assembles the final `phase1_state` dict (`excludedNgrams`/`clusterDefs`) | Phase 1's save cell, `run_pipeline.py` |
| `save_phase1_state(state, path)` | Writes `phase1_state.json` | Same |
| `hex_to_rgb(h)` / `build_cluster_html(final_clusters, corpus_name, ...)` / `save_html(html, path)` | Tableau20-colored HTML cluster summary | Phase 1's HTML export cell, `run_pipeline.py` |

## `phase2/pipeline.py`

Informative-sentence selection (lexical density + stem/n-gram matching).

| Function | Purpose | Called from |
|---|---|---|
| `load_phase1_state(path)` | Loads `phase1_state.json` | Phase 2's load cell, `run_pipeline.py` |
| `segment_corpus(nlp, text)` | Sentence-segments the corpus | `enrich_with_informative_sentences`, `common/parameter_sweep.py::compute_sweep_report` |
| `lexical_density(doc)` | Content-word share of one sentence | `compute_lexical_densities`, `select_representative_sentences` |
| `compute_lexical_densities(sentence_docs, percentile)` | Per-sentence density + the percentile threshold | `enrich_with_informative_sentences`, `common/parameter_sweep.py::compute_sweep_report` |
| `stem_matches(doc, cluster_stems, stemmer)` | Which cluster stems appear in a sentence (stemmer applied to the sentence's own tokens, so inflection-agnostic) | `select_representative_sentences` |
| `_lemma_tokens(doc)` | A sentence's alpha-only, non-stopword tokens as `(lemma.lower(), doc_token_index)` pairs -- mirrors `phase1_pipeline.tokenize_many_for_analysis`'s exact filtering so it's directly comparable to how cluster n-grams were mined | `ngram_matches` |
| `ngram_matches(doc, ngrams)` | Which of a cluster's n-grams (lemma-joined strings, e.g. `"trust human"`) occur in a sentence, matched lemma-to-lemma via `_lemma_tokens` rather than literal text -- fixes a real bug where a `PhraseMatcher` literal-text search almost never found anything, since the n-gram string is a lemma reconstruction, not a corpus quote (verified on real Boisseau data: 17 of 25 cluster n-grams never occurred as that exact literal substring anywhere in the raw text, so a literal search could never match them regardless of sentence selection; all 25 became matchable after this fix). Returns `(matched_ngram_strings, matched_doc_token_indices)` -- the strings are the canonical lemma form (for scoring/display), the indices are the real, verbatim tokens that triggered the match (for highlighting) | `select_representative_sentences` |
| `highlight_sentence(doc, matched_stems, matched_ngram_indices, stemmer)` | Wraps matched spans in `****...****` markers; n-gram indices come pre-resolved from `ngram_matches`, so highlighting always bolds the real inflected words that occurred, even though `matched_ngrams` (elsewhere) reports the canonical lemma form | `select_representative_sentences` |
| `score_sentence(density, matched_stems, matched_ngrams)` | `density * (1 + stems + ngrams)` | `select_representative_sentences` |
| `select_representative_sentences(cluster, sentence_docs, densities, density_threshold, stemmer, top_n)` | Top-N scored sentences for one cluster (or pseudo-cluster -- only ever reads `cluster["stems"]`/`cluster["ngrams"]`, no dependency on real Phase 1 clustering). No longer takes `nlp` -- matching is now lemma-based off the already-parsed `sentence_docs`, not a `PhraseMatcher` built from `nlp.vocab` | `enrich_with_informative_sentences`, `common/parameter_sweep.py::compute_sweep_report` |
| `enrich_with_informative_sentences(phase1_state, text, nlp, stemmer, top_n, density_percentile)` | Orchestrates the above for every cluster. Every interface logs the resolved `top_n`/`density_percentile` right after this call as one `step="phase2_setup"`, `decision_type="sentence_selection_config"` decision (`choice="top_n=X, density_percentile=Y"`) -- previously only the webapp did this; CLI/notebook now match | Phase 2's selection cell, `run_pipeline.py` |
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
| `_split_sentences_loose(text)` | Lightweight regex sentence split (no spaCy dependency) | `_check_invented_token_run` |
| `_check_invented_token_run(condensed_text, ordered_sentences)` | Flags a run of consecutive words (>= the condensation's mean words/sentence) absent from the informative sentences and not an ordinary function word | `check_condensation_sanity` |
| `_check_mega_long_word(condensed_text, max_word_length=)` | Flags a single word over `max_word_length` (default 30) characters | Same |
| `_check_repeated_ngram(condensed_text, n=, min_repeats=)` | Flags an exact `n`-word phrase (default 6) repeated `min_repeats`+ times | Same |
| `check_condensation_sanity(condensed_text, ordered_sentences)` | Runs all three checks above; returns a list of issue descriptions (empty if clean) | `attempt_condensation_trials`, report builders |
| `format_trial_line(t, prefix=)` | One console/log line for a generation trial (word count + any sanity issues) -- shared so `run_pipeline.py`/`webapp` don't re-derive the format | `run_generate_for_rate` and its CLI/webapp equivalents |
| `attempt_condensation_trials(..., model, max_trials, include_full_text, ...)` | Up to `max_trials` generation attempts against Ollama (`repeat_penalty: 1.0`, `think: False`), no `input()`. Each trial is checked against both word-count tolerance and `check_condensation_sanity`; a trial only short-circuits as success when both pass. Returns `{"text", "trials", "success", "soft_accept", "sanity_failed"}` -- falls back to the closest sane trial, or, only if every trial failed sanity, the closest trial overall (tagged `sanity_failed: True`), rather than ever discarding output silently | `run_generate_for_rate` |
| **`run_condensation_setup(decisions, host=)`** (interactive) | Ollama availability check + rate(s)/model/adversarial-reviewer-model (defaults to the generation model on ENTER)/max-trials prompts; returns `(rates, ollama_model, max_trials, adversarial_model)` | phase2.ipynb, `run_pipeline.py::resolve_condensation_config` |
| **`run_generate_for_rate(rate, ..., decisions, host=)`** (interactive) | Non-escalated trials, then the escalation prompt/retry if needed | phase2.ipynb, `run_pipeline.py::run_phase2` |
| `build_adversarial_review_prompt(condensed_text, ordered_sentences, target_words, corpus_name)` | Builds the adversarial-reviewer prompt: checks for an abrupt ending, garbled/nonsensical sequences, or drift outside the informative-sentence pool; asks for a fixed `VERDICT: OK` / `VERDICT: FIXED`+`ISSUES:`+`TEXT:` response | `run_adversarial_review` |
| `parse_adversarial_response(response_text)` | Parses that structured response; returns `(verdict, issues, fixed_text)`, `"UNPARSEABLE"` if the model didn't follow the format | `run_adversarial_review` |
| `run_adversarial_review(condensed_text, ordered_sentences, target_words, corpus_name, adversarial_model, decisions, rate, host=)` | Runs the reviewer once per rate, after a trial is accepted and before it's saved; fails safe (keeps the original text) on an unreachable Ollama, an unparseable response, or a fixed text that's empty/implausibly short (`< 0.5 * target_words`). Always logs `decision_type="adversarial_review_result"` with the original text in `extra` for audit. Returns `(final_text, was_fixed, original_text, issues)` | `run_pipeline.py::run_phase2`/`run_regeneration_loop`, `webapp/pipeline_session.py::_step_injection_analysis`/`_step_regenerate_after_text`, phase2.ipynb |
| `_chunk_source_text(source_text, max_chunk_words=)` | Paragraph-level chunks of the source (further split if over `max_chunk_words`) | `extract_source_metadata` |
| `_parse_source_metadata_response(response)` | Parses the fixed `"TITLE: ...\nAUTHOR(S): ...\nDATE: ..."` LLM response format; missing/blank fields become `"Unclear"` | Same |
| `extract_source_metadata(source_text, model, embedder_name=, top_k=, host=)` | Retrieval-augmented title/author(s)/date extraction: embeds source chunks + a synthetic query, retrieves the `top_k` most relevant chunks by cosine similarity, asks the LLM to extract structured metadata from just those | `run_source_metadata_setup`, webapp's auto-detect path |
| **`run_source_metadata_setup(decisions, source_text, ollama_model, host=)`** (interactive) | Manual entry (blank -> "Unclear") or LLM auto-detection of the source's title/author(s)/date. Like the post-condensation regeneration feature, this is wired into `run_pipeline.py` and the webapp only, not the notebooks | `run_pipeline.py::run_phase2` |
| `in_source(chunk_words, source_lower)` | Whole-phrase verbatim membership test | `classify_span`, `scan_verbatim_overlap` |
| `scan_verbatim_overlap(condensed_text, source_text, ...)` | Longest-run token-alignment scan against the source | `compute_injection_stats` |
| `_content_words(text, extra_function_words=)` | Content (non-function-word) tokens of a string | `classify_span`'s overlap alignment, `flag_borderline_classifications` |
| `is_metalinguistic(sentence)` | Keyword check for "about the text's own argument" sentences | `classify_span` |
| `_is_connective_token(token)` | F's connective test on one spaCy token (`CONNECTIVE_POS`, `dep_=="advmod"`, or freestanding `ADV`+`ROOT`) | `classify_span`, `_is_lexical_token` |
| `_is_lexical_token(token)` | T's genuine-content test on one spaCy token (`LEXICAL_POS` or a wh-word `tag_`), excluding anything `_is_connective_token` already claims | `classify_span` |
| `_diff_inserted_tokens(condensed_tokens, source_word_tokens)` | `difflib`-based diff of a condensed sentence's tokens against its best-matching source sentence's tokens; returns just the inserted/changed tokens (deletions ignored) | `classify_span` |
| `classify_span(sent, source_sentences, source_lower, span_id)` | F/T/R/C classification of one sentence (`sent` is a spaCy `Span`, not a string; `None` if verbatim). Finds the best-matching source sentence, diffs the two, and classifies the whole sentence: F if the delta is <=4 purely connective tokens, T if it's <=6 tokens including genuine lexical content (or a metalinguistic-phrase match), R/C only once both are ruled out. C's `source_refs` lists every source sentence with >=0.15 content-word overlap (sorted strongest-first), not just the single best match. A metalinguistic-phrase T also carries forward `candidate_overlap`/`candidate_diff_text` -- the same best-match overlap score and diff already computed for the F check above it, not thrown away -- so `flag_borderline_classifications` can cross-check by diff instead of only a whole-sentence word count when the candidate is a strong match | `classify_condensation` |
| `classify_condensation(condensed_text, source_text, nlp)` | Classifies every body-prose sentence, block by block | Phase 2's injection-analysis cell, `run_pipeline.py` |
| `flag_borderline_classifications(all_spans)` | Cross-checks F/T's POS/dependency-based verdict against the old lexical word-list heuristic and flags disagreement -- on the diff tokens for diff-based spans and for a phrase-match T with a strong (>=0.7 overlap -- deliberately stricter than R's own 0.5 "dominant match" bar, since a diff-based flag here is only trustworthy against a near-verbatim candidate) candidate source sentence, on the whole sentence only for the no-match fallback and a phrase-match T with no trustworthy candidate to diff against. The phrase-match diff path catches e.g. a metalinguistic-tagged sentence that's actually near-verbatim with one real word changed (real example: flagged as "diverges by 1 content-bearing token" instead of a much less informative "17 content-bearing tokens" for a sentence 94% overlapping its source) | Same |
| **`run_injection_review(all_spans, borderline_flags, rate, decisions)`** (interactive) | Per-flag keep-or-reclassify prompt | phase2.ipynb, `run_pipeline.py::run_phase2` |
| `compute_injection_stats(all_spans, condensed_text, source_text)` | `non_injected_pct` + `verbatim_overlap_pct` (two independent checks) | Phase 2's injection-analysis cell, `run_pipeline.py` |
| `_cluster_hit_counts(text, clusters, stemmer)` | Token/phrase-occurrence hit counts per cluster, for one text | `compute_cluster_coverage` |
| `compute_cluster_coverage(phase1_state, condensed_text, source_text, stemmer, tolerance_pp=)` | Per-cluster share of all cluster-vocabulary occurrences (source vs. condensation), with OK/WARN/DARK status | Phase 2's coverage cell, `run_pipeline.py` |

## `phase2/condensation_report.py`

Renders `condense.py`'s results into an HTML fragment and the plainer
report formats.

| Function | Purpose | Called from |
|---|---|---|
| `parse_condensed_blocks(condensed_text)` | Splits condensed text into typed blocks (`h2`/`h3`/`defn`/`p`). No longer guesses a title/author from the first two lines -- that positional heuristic misfired whenever the condensation didn't open with a literal one-line title (the normal case); the report's title/author/date now come from `run_source_metadata_setup`/`extract_source_metadata` instead | `classify_condensation` (condense.py), `build_condensation_fragment`, `build_plain_summary` |
| `_spans_in_block(block_text, all_spans)` | Which classified spans fall inside one block | `_render_blocks_with_toggles` |
| `render_spans(block_text, all_spans)` | Wraps classified spans in their `S['inj_*']` style | `_render_blocks_with_toggles` |
| `render_c_toggle(span_id, source_texts, max_shown=)` / `render_r_toggle(span_id, source_text)` | Collapsed `<details>` source-reveal toggles; `render_c_toggle` shows the first `max_shown` (default 5) sources directly and nests any remainder behind a second "Show N more" toggle | `_render_blocks_with_toggles` |
| `_render_blocks_with_toggles(body_blocks, all_spans, source_sentences)` | Renders every body block with inline toggles after C/R spans | `build_condensation_fragment` |
| `build_meta_legend(..., adversarial_notice=None)` | Metadata table (+ an "Adversarial review" row when the text was auto-fixed) + F/T/R/C color legend | `build_condensation_fragment` |
| `build_coverage_table(coverage_report)` | Cluster-coverage HTML table | `build_condensation_fragment` |
| `build_condensation_fragment(condensed_text, all_spans, source_sentences, coverage_report, ..., title=, authors=, date=, adversarial_notice=None)` | Assembles the full inline-style HTML fragment: metadata table + F/T/R/C legend first, then the title/byline header (from the explicit `title`/`authors`/`date` args, optional -- falls back to `corpus_name` with no byline), then the body and coverage table. `adversarial_notice` (the issue list from `condense.run_adversarial_review`, if it fixed the text) threads through to `build_meta_legend` | Phase 2's export cell, `run_pipeline.py` (result later embedded as-is by `standalone_report.build_standalone_report`) |
| `build_standalone_preview(fragment_html, corpus_name)` | Wraps the fragment in a minimal browsable `<html>` | Same callers |
| `build_human_report(all_spans, borderline_flags, coverage_report, ..., sanity_issues=, adversarial_notice=None)` | Plain-text verification report, including a "Generation sanity checks" section and an "Adversarial review" section | Same |
| `build_plain_summary(blocks)` | Markup-free title/body text | Same |
| `save_condensation_outputs(paths_dict, fragment, preview, report, summary, injection_report_data)` | Writes 5 of the 6 per-rate output files (condensed text is saved separately) | Phase 2's export cell, `run_pipeline.py` |

---

## `phase3/terms.py`

Shared term-matching utilities: finding a cluster stem/n-gram's token
positions in a document (stems by exact PorterStemmer match, n-grams by
exact lemma match over a contiguous span -- mirroring how
`phase1/pipeline.py` itself builds each), and a proximity check between
two position lists.

| Function | Purpose | Called from |
|---|---|---|
| `token_forms(doc, stemmer)` | Returns `(stems, lemmas)`, one stemmed and one lemmatized string per alpha token | Every other function in this file, `collocations.py`, `distant_reading.py` |
| `term_positions(stems, lemmas, term)` | Token indices where a stem/n-gram occurs | Same callers |
| `count_positions_near(pos_a, pos_b, proximity_n)` | How many `pos_a` entries have a `pos_b` entry within `proximity_n` tokens -- two-pointer sweep | `collocations.py::find_source_collocations` |
| `positions_near(pos_a, target_positions, proximity_n)` | Filters `pos_a` down to entries near any `target_positions` entry | `distant_reading.py::build_contexts_table` |

## `phase3/stopwords.py`

Comprehensive stopword construction for Phase 3's frequency-based tools.

| Function | Purpose | Called from |
|---|---|---|
| `numeral_stopwords(text)` | Numeral tokens (incl. `2024a`-style) | `build_stopwords` |
| `candidate_author_surnames(doc)` | PERSON-entity names, ranked by mention count | `build_stopwords`, report provenance |
| `build_stopwords(nlp_stopwords, doc, min_author_mentions=)` | spaCy's stopword list + numerals + citation abbreviations + auto-included author names (3+ mentions, disclosed, never researcher-confirmed) | `phase3/pipeline.py::prepare_phase3_context` |

## `phase3/collocations.py`

Empirical collocation-pair discovery: scans the source for term pairs
(from different clusters) that actually co-occur, ranks by hit count,
flags base-rate confounds.

| Function | Purpose | Called from |
|---|---|---|
| `find_source_collocations(doc, clusterdefs, stemmer, ...)` | Ranked, confound-flagged candidate list | `phase3/pipeline.py::prepare_phase3_context` |
| `format_collocation_candidates(candidates)` | Console-printable candidate list | `run_collocation_review` |
| `select_pairs_by_index(candidates, indices)` | Resolves 0-based indices to candidates, falling back to the top-ranked one | `run_collocation_review`, `webapp/pipeline_session.py::apply_collocation_review` |
| **`run_collocation_review(candidates, decisions)`** (interactive) | Prints candidates, prompts for a selection -- comma-separated indices, `all`/`*` for every candidate, or ENTER for the top-ranked one; always logs the choice | `run_pipeline.py::run_phase3` |

## `phase3/distant_reading.py`

Single-corpus analyses, reused once per document by `comparison.py` --
Python-native reimplementations of a retired set of Voyant tool cells
(Reader, Cirrus, Trends, Phrases, CorpusTerms, Contexts; CollocatesGraph
is replaced by a co-occurrence table, Bubblelines is not reproduced
separately from Trends).

| Function | Purpose | Called from |
|---|---|---|
| `build_reader_html(doc, clusterdefs, colors_by_cluster, stemmer, ...)` | Full text with cluster terms highlighted, token-level matching. A stem/n-gram claimed by more than one cluster (a real possibility -- see recluster_noise) is highlighted with every claiming cluster's color, striped together (`_mark_background`), tooltip listing every cluster name -- not silently attributed to whichever cluster comes first alphabetically | `phase3/pipeline.py::build_phase3_report` |
| `build_wordcloud_html(text, stopwords, ...)` | Word cloud PNG (via `wordcloud`), embedded as a base64 `<img>` | `build_phase3_report`, `comparison.py` |
| `bin_cluster_frequencies(text, clusterdefs, stemmer, n_bins=)` | Per-cluster hit counts across `n_bins` segments (reuses `condense.py::_cluster_hit_counts`) | `build_phase3_report` |
| `build_trend_chart_svg(bin_freqs, colors_by_cluster, ...)` | Hand-rolled inline SVG line chart, no dependency | `build_phase3_report` |
| `build_phrase_table(doc, stopwords, min_n=, max_n=, top_n=)` | N-gram frequency table, overlap-filtered | `build_phase3_report` |
| `build_term_stats_table(doc, clusterdefs, stemmer, n_bins=, top_n=)` | Raw/relative frequency + peakedness (`scipy.stats.kurtosis`) + skewness (`scipy.stats.skew`) per term, with a sparkline | `build_phase3_report` |
| `build_contexts_table(doc, stemmer, term_a, term_b=, ...)` | KWIC concordance, optionally restricted to occurrences near `term_b` | `build_phase3_report`, `comparison.py` |
| `build_collocates_table(doc, stemmer, anchor_term, ...)` | Co-occurrence table for one anchor term | `comparison.py` |

## `phase3/comparison.py`

Source-vs-Summary comparison, built once per approved condensation rate.

| Function | Purpose | Called from |
|---|---|---|
| `build_document_profile_table(documents)` | Word count/unique words/lexical density per document | `build_comparison_section` |
| `build_comparison_section(source_doc, rate_docs, phase1_state, source_text, stemmer, stopwords, selected_pairs)` | Assembles the whole "Source vs. Summary" section -- document profile, cluster coverage (reuses `condense.compute_cluster_coverage`/`condensation_report.build_coverage_table`), word clouds, Contexts, Collocates | `phase3/pipeline.py::build_phase3_report` |

## `phase3/report.py`

Assembles the standalone HTML report, same inline-style convention as
`common/standalone_report.py`.

| Function | Purpose | Called from |
|---|---|---|
| `build_cluster_legend_html(clusterdefs, colors_by_cluster)` | Colour-coded cluster legend table | `build_distant_reading_report` |
| `build_cross_cluster_html(clusterdefs)` | Stems appearing in more than one cluster | `build_distant_reading_report` |
| `build_provenance_html(stopword_count, auto_authors, collocation_candidates, selected_pairs)` | Discloses stopword/author/collocation-selection provenance | `phase3/pipeline.py::build_phase3_report` |
| `build_distant_reading_report(...)` | Assembles the full standalone HTML page | `phase3/pipeline.py::build_phase3_report` |
| `save_distant_reading_report(html, path)` | Writes the report to disk | `run_pipeline.py`, `webapp/pipeline_session.py` |

## `phase3/pipeline.py`

Top-level, non-interactive Phase 3 orchestration -- mirrors
`phase1/pipeline.py`'s split between pure functions and CLI-only
`run_*` wrappers.

| Function | Purpose | Called from |
|---|---|---|
| `assign_cluster_colors(clusterdefs, tableau20=)` | Tableau20 colours, sequential by cluster order | `prepare_phase3_context` |
| `prepare_phase3_context(phase1_state, source_text, nlp, stemmer)` | Parses the source, builds stopwords, scans for collocations -- everything needed before the one researcher decision | `run_pipeline.py::run_phase3`, `webapp/pipeline_session.py::_step_prepare_phase3` |
| `build_phase3_report(context, corpus_name, phase1_state, source_text, stemmer, condensed_texts, selected_pairs, nlp, n_bins=)` | Builds every section and returns the report HTML; `condensed_texts={}` skips just the Source-vs-Summary section | Same callers |

---

## `run_pipeline.py` (repo root)

Runs Phase 0 (optional) -> Phase 1 -> Phase 2 -> Phase 3 -> standalone
report export on one new corpus in a single script -- see README's
"Running the pipeline" for usage.

| Function | Purpose |
|---|---|
| `parse_args()` | CLI flags: `--corpus`, `--input`, `--clean`, `--start-phase` (`1`/`2`/`3`, default `1`), `--list-corpora`, one flag per notebook config constant, `--merge-duplicate-word-occurrences`, `--sweep` (Phase 1 parameter-sweep opt-in), optional `--rates`/`--ollama-model`/`--adversarial-model`/`--max-trials`. `--corpus`/`--input` are no longer unconditionally `required=True` -- validated in `main()` instead, since whether `--input` is required (and `--corpus` at all, given `--list-corpora`) depends on `--start-phase` |
| `place_raw_text(args, paths)` | Copies `--input` to `paths.raw_txt()`, or runs `clean_corpus.clean_text` first if `--clean`. Only called for `--start-phase 1` |
| `resolve_condensation_config(args, decisions)` | Uses CLI condensation flags if `--rates`/`--ollama-model`/`--max-trials` are all given (`--adversarial-model` optional even then, defaults to `--ollama-model`; logged with `"source": "cli"`); else falls back to `condense.run_condensation_setup` |
| `run_phase1(args, paths, decisions)` | Mirrors `phase1.ipynb` cell-by-cell -- including, if `--sweep` is given, `common.parameter_sweep.run_parameter_sweep_setup` right before stem extraction; returns `(phase1_state, nlp, stemmer, text)` |
| `run_phase2(args, paths, decisions, phase1_state, nlp, stemmer, text, original_text)` | Mirrors `phase2.ipynb` cell-by-cell, plus `condense.run_source_metadata_setup`, `condense.run_adversarial_review` per rate (right before `process_and_save_rate`), `run_phase3` (right after the initial rates' reports are built), and the post-completion regeneration loop (`run_regeneration_loop`) -- none of which the notebook has. Reached from `--start-phase` 1 or 2 |
| `run_phase3(args, paths, phase1_state, nlp, stemmer, text, condensed_texts)` | No notebook equivalent. Builds the Phase 3 context, runs `collocations.run_collocation_review` if there's a condensation to compare against, and writes the distant-reading report -- own `DecisionLog(phase="phase3")`. Reached from `--start-phase` 1, 2 (via `run_phase2`), or 3 (called directly) |
| `_print_corpus_list()` | Backs `--list-corpora`: prints every corpus from `resume.list_corpora()` with `resume.corpus_phase_summary()`'s resumable phases and condensed rates, then returns without running anything |
| `_load_resume_state(args, paths)` | Backs `--start-phase 2`/`3`: loads `text`/`phase1_state`/`nlp`/`stemmer` from the existing corpus on disk instead of producing them via `place_raw_text`/`run_phase1`. `lang_model` is recovered from Phase 1's own decision log via `resume.find_last_decision_choice` if available, else `--lang-model`'s default |
| `main()` | Branches on `--start-phase`: `1` wires `place_raw_text` -> `run_phase1` -> `run_phase2` as before; `2` calls `_load_resume_state` then `run_phase2` directly (with `original_text` falling back to the saved raw text, since the true pre-clean original is never persisted); `3` calls `_load_resume_state`, reconstructs `condensed_texts` from every existing condensed-rate file on disk, then calls `run_phase3` directly. `resume.check_prerequisites` gates `--start-phase 2`/`3` before `paths.ensure_dirs()` runs -- checked first so a failed check doesn't leave behind empty `data/<corpus>/*` directories |
