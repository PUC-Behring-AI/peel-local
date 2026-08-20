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
| `generate(model, prompt, host=, timeout=)` | Blocking, non-streaming `/api/generate` call; raises `OllamaError` on failure | `condense.py::attempt_condensation_trials` |

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
| `glossbert_predict(occurrence, tokenizer, model, device, ...)` | Scores a word occurrence against its candidate WordNet senses; marks the target word in its sentence via a case-insensitive regex search (robust when `word` is lowercase but `sentence` retains original casing, e.g. an all-caps heading) | `run_glossbert_analysis`, `recluster_large_clusters`, `recluster_noise` |
| `extract_top_stems(doc, stemmer, top_percentile, max_stems)` | Frequency-ranked stem extraction | Phase 1's stem-extraction cell, `run_pipeline.py` |
| `map_stems_to_sentences(doc, top_stems, stemmer)` | Builds `{stem: [occurrence, ...]}` | Phase 1's GlossBERT cell, `run_pipeline.py` |
| `record_accepted_instance(accepted_definitions, stem, word, pos, definition, sentence, count)` | Insert-or-append into `accepted_definitions[stem]` | `run_glossbert_analysis`, `run_flagged_term_review` |
| `run_glossbert_analysis(top_stems, stem_occurrences, tokenizer, model, device, ...)` | Runs WSD per stem (words lowercased, `PROPN` mapped to `wn.NOUN`); flags sense mismatches, deduped per stem by `(word, default_sense, predicted_sense)` so e.g. the same word from an all-caps heading and from normal prose only produces one review item; `top_candidates` respects `max_synsets` (previously hardcoded to 3) | Phase 1's GlossBERT cell, `run_pipeline.py` |
| `resolve_flagged_choice(item, choice)` | Pure resolution of one flagged-term review choice (no `input()`); candidate slots and the manual-entry slot are keyed off `len(item["top_candidates"])`, not a fixed count | `run_flagged_term_review` |
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
| **`run_cluster_review(renamed_clusters, decisions)`** (interactive) | Per-cluster accept/rename/remove-stems/remove-ngrams flow | phase1.ipynb, `run_pipeline.py::run_phase1` |
| `build_phase1_state(final_clusters, excluded_cluster_ngrams)` | Assembles the final `phase1_state` dict (`excludedNgrams`/`clusterDefs`) | Phase 1's save cell, `run_pipeline.py` |
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
| `_split_sentences_loose(text)` | Lightweight regex sentence split (no spaCy dependency) | `_check_invented_token_run` |
| `_check_invented_token_run(condensed_text, ordered_sentences)` | Flags a run of consecutive words (>= the condensation's mean words/sentence) absent from the informative sentences and not an ordinary function word | `check_condensation_sanity` |
| `_check_mega_long_word(condensed_text, max_word_length=)` | Flags a single word over `max_word_length` (default 30) characters | Same |
| `_check_repeated_ngram(condensed_text, n=, min_repeats=)` | Flags an exact `n`-word phrase (default 6) repeated `min_repeats`+ times | Same |
| `check_condensation_sanity(condensed_text, ordered_sentences)` | Runs all three checks above; returns a list of issue descriptions (empty if clean) | `attempt_condensation_trials`, report builders |
| `format_trial_line(t, prefix=)` | One console/log line for a generation trial (word count + any sanity issues) -- shared so `run_pipeline.py`/`webapp` don't re-derive the format | `run_generate_for_rate` and its CLI/webapp equivalents |
| `attempt_condensation_trials(..., model, max_trials, include_full_text, ...)` | Up to `max_trials` generation attempts against Ollama (`repeat_penalty: 1.0`, `think: False`), no `input()`. Each trial is checked against both word-count tolerance and `check_condensation_sanity`; a trial only short-circuits as success when both pass. Returns `{"text", "trials", "success", "soft_accept", "sanity_failed"}` -- falls back to the closest sane trial, or, only if every trial failed sanity, the closest trial overall (tagged `sanity_failed: True`), rather than ever discarding output silently | `run_generate_for_rate` |
| **`run_condensation_setup(decisions, host=)`** (interactive) | Ollama availability check + rate(s)/model/max-trials prompts | phase2.ipynb, `run_pipeline.py::resolve_condensation_config` |
| **`run_generate_for_rate(rate, ..., decisions, host=)`** (interactive) | Non-escalated trials, then the escalation prompt/retry if needed | phase2.ipynb, `run_pipeline.py::run_phase2` |
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
| `classify_span(sent, source_sentences, source_lower, span_id)` | F/T/R/C classification of one sentence (`sent` is a spaCy `Span`, not a string; `None` if verbatim). Finds the best-matching source sentence, diffs the two, and classifies the whole sentence: F if the delta is <=4 purely connective tokens, T if it's <=6 tokens including genuine lexical content (or a metalinguistic-phrase match), R/C only once both are ruled out. C's `source_refs` lists every source sentence with >=0.15 content-word overlap (sorted strongest-first), not just the single best match | `classify_condensation` |
| `classify_condensation(condensed_text, source_text, nlp)` | Classifies every body-prose sentence, block by block | Phase 2's injection-analysis cell, `run_pipeline.py` |
| `flag_borderline_classifications(all_spans)` | Cross-checks F/T's POS/dependency-based verdict against the old lexical word-list heuristic and flags disagreement (on the diff tokens for diff-based spans, the whole sentence for the no-match/phrase-match fallbacks) | Same |
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
| `build_meta_legend(...)` | Metadata table + F/T/R/C color legend | `build_condensation_fragment` |
| `build_coverage_table(coverage_report)` | Cluster-coverage HTML table | `build_condensation_fragment` |
| `build_condensation_fragment(condensed_text, all_spans, source_sentences, coverage_report, ..., title=, authors=, date=)` | Assembles the full inline-style HTML fragment: metadata table + F/T/R/C legend first, then the title/byline header (from the explicit `title`/`authors`/`date` args, optional -- falls back to `corpus_name` with no byline), then the body and coverage table | Phase 2's export cell, `run_pipeline.py` (result later embedded as-is by `standalone_report.build_standalone_report`) |
| `build_standalone_preview(fragment_html, corpus_name)` | Wraps the fragment in a minimal browsable `<html>` | Same callers |
| `build_human_report(all_spans, borderline_flags, coverage_report, ..., sanity_issues=)` | Plain-text verification report, including a "Generation sanity checks" section | Same |
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
| **`run_collocation_review(candidates, decisions)`** (interactive) | Prints candidates, prompts for a selection | `run_pipeline.py::run_phase3` |

## `phase3/distant_reading.py`

Single-corpus analyses, reused once per document by `comparison.py` --
Python-native reimplementations of a retired set of Voyant tool cells
(Reader, Cirrus, Trends, Phrases, CorpusTerms, Contexts; CollocatesGraph
is replaced by a co-occurrence table, Bubblelines is not reproduced
separately from Trends).

| Function | Purpose | Called from |
|---|---|---|
| `build_reader_html(doc, clusterdefs, colors_by_cluster, stemmer, ...)` | Full text with cluster terms highlighted, token-level matching | `phase3/pipeline.py::build_phase3_report` |
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
| `parse_args()` | CLI flags: `--corpus`, `--input`, `--clean`, one flag per notebook config constant, optional `--rates`/`--ollama-model`/`--max-trials` |
| `place_raw_text(args, paths)` | Copies `--input` to `paths.raw_txt()`, or runs `clean_corpus.clean_text` first if `--clean` |
| `resolve_condensation_config(args, decisions)` | Uses CLI condensation flags if all three given (logged with `"source": "cli"`); else falls back to `condense.run_condensation_setup` |
| `run_phase1(args, paths, decisions)` | Mirrors `phase1.ipynb` cell-by-cell; returns `(phase1_state, nlp, stemmer, text)` |
| `run_phase2(args, paths, decisions, phase1_state, nlp, stemmer, text)` | Mirrors `phase2.ipynb` cell-by-cell, plus `condense.run_source_metadata_setup`, `run_phase3` (right after the initial rates' reports are built), and the post-completion regeneration loop (`run_regeneration_loop`) -- none of which the notebook has |
| `run_phase3(args, paths, phase1_state, nlp, stemmer, text, condensed_texts)` | No notebook equivalent. Builds the Phase 3 context, runs `collocations.run_collocation_review` if there's a condensation to compare against, and writes the distant-reading report -- own `DecisionLog(phase="phase3")` |
| `main()` | Wires `CorpusPaths`, `place_raw_text`, `run_phase1` (its own `DecisionLog(phase="phase1")`), `run_phase2` (its own `DecisionLog(phase="phase2")`) |
