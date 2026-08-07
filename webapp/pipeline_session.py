"""PipelineSession: the web UI's orchestration layer.

This is a fourth way to drive the same pipeline the notebooks and
run_pipeline.py already run -- it calls the exact same functions in
phase1/pipeline.py, phase2/pipeline.py, phase2/condense.py,
phase2/condensation_report.py, common/voyant_notebook.py,
common/standalone_report.py. Nothing about the pipeline itself is
reimplemented here.

What IS new here: the interactive review loops in phase1/pipeline.py and
phase2/condense.py (run_flagged_term_review, run_cluster_review, ...) are
built around blocking input() and aren't reusable from a web request.
PipelineSession instead calls the lower-level pure functions they wrap
(record_accepted_instance, attempt_condensation_trials,
classify_condensation, ...) directly, applying decisions submitted as
structured JSON from the browser instead of asking input(). Every
decision is still logged via the existing DecisionLog, tagged
extra={"source": "web"} (same precedent as run_pipeline.py's
extra={"source": "cli"}).

Execution model: each pipeline chunk runs in a background thread (heavy
steps -- GlossBERT, clustering, Ollama generation -- can take minutes),
with stdout captured into a growing log the frontend polls. A chunk
either finishes the whole run (returns None) or returns a dict describing
the next decision the browser needs to collect
({"type": ..., "payload": ...}). Only one PipelineSession runs at a time
(a local, single-researcher tool, not a multi-user server) -- that's what
makes a plain sys.stdout redirect safe without a real logging framework.
"""

import threading
import traceback
from contextlib import redirect_stdout

import spacy
import torch
from nltk.stem import PorterStemmer

from common.paths import CorpusPaths, resources_dir
from common.decisions import DecisionLog
from common import voyant_notebook, standalone_report
from phase0.clean_corpus import clean_text
from phase1 import pipeline as phase1_pipeline
from phase2 import pipeline as phase2_pipeline, condense, condensation_report


class _TeeStream:
    """Appends every write() to session.log while still echoing to the
    real stdout, so the server console shows the same output as the
    browser's log panel."""

    def __init__(self, session, real_stdout):
        self._session = session
        self._real = real_stdout

    def write(self, s):
        if s:
            self._session._log_parts.append(s)
        return self._real.write(s)

    def flush(self):
        self._real.flush()


class PipelineSession:
    def __init__(self):
        self.status = "idle"  # idle | running | awaiting_input | error | complete
        self.current_step = ""
        self.decision = None  # {"type": ..., "payload": ...} when awaiting_input
        self.error = None
        self.manifest = None
        self.corpus_name = None
        self._log_parts = []
        self._regen_rate = None  # set while a post-completion regeneration is in flight

    # ------------------------------------------------------------
    # Status / log access (safe to call from the Flask request thread
    # while a background thread is writing -- CPython's GIL makes these
    # simple reads/appends atomic enough for a local single-user tool)
    # ------------------------------------------------------------

    def status_dict(self, since: int = 0) -> dict:
        log_text = "".join(self._log_parts)
        return {
            "status": self.status,
            "current_step": self.current_step,
            "corpus_name": self.corpus_name,
            "log": log_text[since:],
            "log_length": len(log_text),
            "decision": self.decision,
            "error": self.error,
            "manifest": self.manifest if self.status == "complete" else None,
        }

    # ------------------------------------------------------------
    # Background job runner
    # ------------------------------------------------------------

    def _run_bg(self, target, current_step: str = ""):
        if self.status == "running":
            raise RuntimeError("A step is already running for this session.")

        self.status = "running"
        self.current_step = current_step or getattr(target, "__name__", "")
        self.error = None
        self.decision = None

        def worker():
            import sys as _sys
            real_stdout = _sys.stdout
            tee = _TeeStream(self, real_stdout)
            try:
                with redirect_stdout(tee):
                    result = target()
                if result is None:
                    self.status = "complete"
                    self.decision = None
                else:
                    self.status = "awaiting_input"
                    self.decision = result
            except Exception:
                tb = traceback.format_exc()
                self._log_parts.append("\n" + tb)
                self.error = tb
                self.status = "error"

        threading.Thread(target=worker, daemon=True).start()

    # ------------------------------------------------------------
    # ENTRY POINT
    # ------------------------------------------------------------

    def start(self, corpus_name, file_storage, clean: bool, config: dict):
        """config: dict with top_percentile, max_stems, max_sentences_per_stem,
        max_synsets, max_cluster_size, min_clusters, min_cluster_len,
        glossbert_model, lang_model, sentence_embedder (already typed by app.py)."""
        self.corpus_name = corpus_name
        self.paths = CorpusPaths(corpus_name)
        self.paths.ensure_dirs()
        self.decisions1 = DecisionLog(corpus_name, phase="phase1")
        self.decisions2 = DecisionLog(corpus_name, phase="phase2")
        self.config = config

        if clean:
            raw_text = file_storage.read().decode("utf-8")
            cleaned = clean_text(raw_text)
            with open(self.paths.raw_txt(), "w", encoding="utf-8") as f:
                f.write(cleaned)
        else:
            file_storage.save(str(self.paths.raw_txt()))

        self._run_bg(self._step_start_to_flagged_review, "Extracting stems & running GlossBERT analysis")

    # ------------------------------------------------------------
    # PHASE 1: stems -> GlossBERT -> flagged-term review
    # ------------------------------------------------------------

    def _step_start_to_flagged_review(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.nlp = spacy.load(self.config["lang_model"])
        self.stemmer = PorterStemmer()

        with open(self.paths.raw_txt(), "r", encoding="utf-8") as f:
            self.text = f.read()
        self.doc = self.nlp(self.text)

        self.top_stems = phase1_pipeline.extract_top_stems(
            self.doc, self.stemmer, self.config["top_percentile"], self.config["max_stems"]
        )
        print(f"Selected {len(self.top_stems)} top stems")

        with open(self.paths.top_stems(), "w", encoding="utf-8") as out:
            out.write("stem\tcount\n")
            for stem, count in self.top_stems.items():
                out.write(f"{stem}\t{count}\n")
        print(f"Saved results to {self.paths.top_stems()}")

        print("Loading GlossBERT checkpoint...")
        self.tokenizer, self.model = phase1_pipeline.load_glossbert(self.config["glossbert_model"], self.device)
        print("GlossBERT loaded successfully.")

        self.stem_occurrences = phase1_pipeline.map_stems_to_sentences(self.doc, self.top_stems, self.stemmer)

        print("Running GlossBERT analysis...")
        self.accepted_definitions, self.flagged_words = phase1_pipeline.run_glossbert_analysis(
            self.top_stems, self.stem_occurrences, self.tokenizer, self.model, self.device,
            max_sentences_per_stem=self.config["max_sentences_per_stem"], max_synsets=self.config["max_synsets"],
        )
        print(f"Done. {len(self.flagged_words)} flagged term(s) need review.")

        if not self.flagged_words:
            print("No flagged terms found -- skipping straight to cluster review.")
            return self._step_after_flagged_review()

        return {
            "type": "flagged_term_review",
            "payload": {
                "items": [
                    {
                        "id": i,
                        "word": item["word"], "stem": item["stem"], "pos": item["pos"],
                        "count": item["count"], "sentence": item["sentence"],
                        "default_sense": item["default_sense"], "default_definition": item["default_definition"],
                        "predicted_sense": item["predicted_sense"], "predicted_definition": item["predicted_definition"],
                        "top_candidates": item["top_candidates"],
                    }
                    for i, item in enumerate(self.flagged_words)
                ]
            },
        }

    def apply_flagged_term_review(self, decisions_payload):
        """decisions_payload: list of {id, choice, candidate_index?, manual_definition?}
        choice in "default" | "predicted" | "candidate" | "manual"."""
        self.decisions1.record(
            step="flagged_term_review", decision_type="batch_review",
            prompt="Flagged term review (web batch)",
            choice=f"{len(decisions_payload)} item(s) reviewed",
            extra={"source": "web", "decisions": decisions_payload},
        )

        for entry in decisions_payload:
            item = self.flagged_words[entry["id"]]
            choice = entry["choice"]
            if choice == "default":
                definition = item["default_definition"]
            elif choice == "predicted":
                definition = item["predicted_definition"]
            elif choice == "candidate":
                definition = item["top_candidates"][entry["candidate_index"]]["definition"]
            elif choice == "manual":
                definition = entry["manual_definition"]
            else:
                raise ValueError(f"Unknown flagged-term choice: {choice!r}")

            phase1_pipeline.record_accepted_instance(
                self.accepted_definitions, item["stem"], item["word"], item["pos"],
                definition, item["sentence"], item["count"],
            )

        self._run_bg(self._step_after_flagged_review, "Clustering accepted definitions")

    # ------------------------------------------------------------
    # PHASE 1: embeddings -> clustering -> reclustering -> ngrams -> cluster review
    # ------------------------------------------------------------

    def _step_after_flagged_review(self):
        print("Saving updated results...")
        phase1_pipeline.save_glossbert_output(self.accepted_definitions, self.paths.glossbert_output())
        print(f"Updated results saved to {self.paths.glossbert_output()}")

        self.embedder, _stem_texts, self.stem_names, embeddings = phase1_pipeline.build_stem_embeddings(
            self.accepted_definitions, self.config["sentence_embedder"]
        )
        self.labels, self.clusters = phase1_pipeline.cluster_stem_embeddings(
            embeddings, self.stem_names, self.config["min_clusters"],
            min_cluster_len=self.config["min_cluster_len"],
        )
        print(f"Stems embedded: {len(self.stem_names)}")
        print(f"Clusters found (excluding noise): {len(self.clusters)}")

        stem_word_frequencies = phase1_pipeline.build_stem_word_frequency_table(self.stem_occurrences)
        self.clusters, large_subclusters = phase1_pipeline.recluster_large_clusters(
            self.clusters, self.stem_occurrences, self.embedder, self.tokenizer, self.model, self.device,
            max_cluster_size=self.config["max_cluster_size"], min_clusters=self.config["min_clusters"],
            min_cluster_len=self.config["min_cluster_len"], max_synsets=self.config["max_synsets"],
        )
        print(f"{len(large_subclusters)} large-cluster subcluster(s) found.")

        noise_stems = phase1_pipeline.extract_noise_stems(self.stem_names, self.labels)
        self.noise_clusters = phase1_pipeline.recluster_noise(
            noise_stems, self.stem_occurrences, self.embedder, self.tokenizer, self.model, self.device,
            min_clusters=self.config["min_clusters"], min_cluster_len=self.config["min_cluster_len"],
            max_synsets=self.config["max_synsets"],
        )
        print(f"{len(self.noise_clusters)} noise cluster(s) recovered.")

        stem_word_frequencies = phase1_pipeline.build_stem_word_frequency_table(self.stem_occurrences)
        renamed_clusters = phase1_pipeline.rename_clusters(self.clusters, stem_word_frequencies)
        renamed_noise_clusters = phase1_pipeline.rename_clusters(self.noise_clusters, stem_word_frequencies)
        renamed_clusters = phase1_pipeline.merge_named_clusters(renamed_clusters, renamed_noise_clusters)

        sentence_cache = phase1_pipeline.build_sentence_token_cache(
            self.nlp, self.accepted_definitions, self.stemmer, use_lemmas=True
        )
        global_ngram_counter, percentile_threshold = phase1_pipeline.build_global_ngram_statistics(sentence_cache)
        all_clusters = {**self.clusters, **self.noise_clusters}
        cluster_ngrams = phase1_pipeline.extract_cluster_ngrams(
            all_clusters, self.accepted_definitions, sentence_cache,
            global_ngram_counter, percentile_threshold, self.stemmer, use_lemmas=True,
        )
        self.renamed_clusters = phase1_pipeline.attach_ngrams(renamed_clusters, cluster_ngrams)
        print(f"{len(self.renamed_clusters)} named cluster(s) ready for review.")

        return {
            "type": "cluster_review",
            "payload": {
                "clusters": [
                    {"name": name, "stems": data["stems"], "ngrams": data.get("ngrams", [])}
                    for name, data in sorted(self.renamed_clusters.items())
                ]
            },
        }

    def apply_cluster_review(self, entries):
        """entries: list of {original_name, removed?, new_name, kept_stems,
        removed_stems, kept_ngrams, removed_ngrams, globally_excluded_stems}.
        An entry with removed=true is dropped entirely -- it never appears
        in final_clusters (mirrors the CLI's "d = drop entirely" option),
        though any of its stems the user chose to globally exclude still
        land in excluded_cluster_stems."""
        self.decisions1.record(
            step="cluster_review", decision_type="batch_review",
            prompt="Cluster review (web batch)",
            choice=f"{len(entries)} cluster(s) reviewed",
            extra={"source": "web", "decisions": entries},
        )

        final_clusters = {}
        excluded_cluster_stems = set()
        excluded_cluster_ngrams = set()

        # From every original cluster, not just the ones kept below -- a
        # stem stays a candidate for incList even if its cluster was
        # entirely removed, unless explicitly globally excluded.
        all_original_stems = set()
        for cluster_data in self.renamed_clusters.values():
            all_original_stems.update(cluster_data["stems"])

        for entry in entries:
            excluded_cluster_stems.update(entry.get("globally_excluded_stems", []))

            if entry.get("removed"):
                continue

            new_name = (entry.get("new_name") or "").strip() or entry["original_name"]
            excluded_cluster_ngrams.update(entry.get("removed_ngrams", []))

            final_clusters[new_name] = {
                "stems": entry["kept_stems"],
                "ngrams": entry.get("kept_ngrams", []),
                "excluded_stems": entry.get("removed_stems", []),
                "excluded_ngrams": entry.get("removed_ngrams", []),
            }

        self.final_clusters = final_clusters
        self.excluded_cluster_stems = excluded_cluster_stems
        self.excluded_cluster_ngrams = excluded_cluster_ngrams
        self.all_original_stems = all_original_stems

        self._run_bg(
            lambda: {
                "type": "voyant_settings",
                "payload": {
                    "cluster_count": len(final_clusters),
                    "stem_count": len(all_original_stems - excluded_cluster_stems),
                },
            },
            "Preparing Voyant settings",
        )

    # ------------------------------------------------------------
    # PHASE 1: Voyant settings -> finish & save
    # ------------------------------------------------------------

    def apply_voyant_settings(self, corpus_id: str, use_smart_stopwords: bool):
        self.decisions1.record(
            step="voyant_settings", decision_type="corpus_id_entry",
            prompt="Enter Voyant Corpus ID", choice=corpus_id, extra={"source": "web"},
        )
        self.decisions1.record(
            step="voyant_settings", decision_type="stopword_toggle",
            prompt="Use Voyant en_smart stopwords?", options=["y", "n"],
            choice="y" if use_smart_stopwords else "n", extra={"source": "web"},
        )

        self.corpus_id = corpus_id
        self.use_smart_stopwords = use_smart_stopwords
        self.smart_stopwords = (
            phase1_pipeline.read_smart_stopwords(resources_dir() / "stop.en.smart.txt")
            if use_smart_stopwords else []
        )

        self._run_bg(self._step_finish_phase1, "Finishing Phase 1")

    def _step_finish_phase1(self):
        self.phase1_state = phase1_pipeline.build_phase1_state(
            corpus_id=self.corpus_id,
            all_original_stems=self.all_original_stems,
            excluded_cluster_stems=self.excluded_cluster_stems,
            excluded_cluster_ngrams=self.excluded_cluster_ngrams,
            final_clusters=self.final_clusters,
            smart_stopwords=self.smart_stopwords,
            use_smart_stopwords=self.use_smart_stopwords,
        )
        phase1_pipeline.save_phase1_state(self.phase1_state, self.paths.phase1_state_json())
        print(f"Saved Phase 1 state to {self.paths.phase1_state_json()}")

        html = phase1_pipeline.build_cluster_html(self.final_clusters, self.corpus_name)
        phase1_pipeline.save_html(html, self.paths.phase1_html())
        print(f"HTML cluster summary written to {self.paths.phase1_html()}")

        return {
            "type": "phase2_setup",
            "payload": {
                "cluster_count": len(self.final_clusters),
                "phase1_state_path": str(self.paths.phase1_state_json()),
            },
        }

    # ------------------------------------------------------------
    # PHASE 2: sentence selection -> condensation generation
    # ------------------------------------------------------------

    def apply_phase2_setup(self, top_n, density_percentile, run_condensation,
                            rates_str, ollama_model, max_trials):
        self.decisions2.record(
            step="phase2_setup", decision_type="sentence_selection_config",
            prompt="Phase 2 sentence-selection config",
            choice=f"top_n={top_n}, density_percentile={density_percentile}",
            extra={"source": "web"},
        )

        self.top_n = top_n
        self.density_percentile = density_percentile
        self.run_condensation = run_condensation
        self.rates = []

        if run_condensation:
            self.rates = [int(r.strip()) for r in rates_str.split(",") if r.strip()]
            self.ollama_model = ollama_model
            self.max_trials = max_trials

            self.decisions2.record(
                step="condensation_setup", decision_type="rate_selection",
                prompt="What condensation rate(s) would you like? (5-30%)",
                choice=rates_str, extra={"parsed_rates": self.rates, "source": "web"},
            )
            self.decisions2.record(
                step="condensation_setup", decision_type="ollama_model_choice",
                prompt="Which Ollama model would you like to use?",
                choice=ollama_model, extra={"source": "web"},
            )
            self.decisions2.record(
                step="condensation_setup", decision_type="max_trials_choice",
                prompt="How many generation trials before asking to escalate?",
                choice=str(max_trials), extra={"source": "web"},
            )

        self._run_bg(self._step_select_sentences_and_generate, "Selecting informative sentences")

    def _rel(self, path) -> str:
        """Path relative to this corpus's data/ root, forward-slashed --
        what the browser needs for GET /api/files/<corpus>/<subpath>,
        rather than a full filesystem path it has no way to interpret."""
        return path.resolve().relative_to(self.paths.root.resolve()).as_posix()

    def _empty_manifest(self):
        return {
            "phase1_state": self._rel(self.paths.phase1_state_json()),
            "phase1_html": self._rel(self.paths.phase1_html()),
            "phase1_decisions_log": self._rel(self.paths.decisions_log("phase1")),
            "phase2_decisions_log": self._rel(self.paths.decisions_log("phase2")),
            "phase2_output_json": self._rel(self.paths.phase2_output_json()),
            "rates": {},
        }

    def _step_select_sentences_and_generate(self):
        self.enriched_state = phase2_pipeline.enrich_with_informative_sentences(
            self.phase1_state, self.text, self.nlp, self.stemmer,
            top_n=self.top_n, density_percentile=self.density_percentile,
        )
        phase2_pipeline.save_informative_sentences(self.enriched_state, self.paths.phase2_output_json())
        print(f"Saved informative sentences to {self.paths.phase2_output_json()}")

        if not self.run_condensation or not self.rates:
            self.manifest = self._empty_manifest()
            print("Pipeline complete (no condensation requested).")
            return None

        self.ordered_sentences = condense.gather_ordered_informative_sentences(self.enriched_state)
        self.cluster_key_terms = condense.gather_cluster_key_terms(self.enriched_state)

        self.condensed_texts = {}
        self._target_words_by_rate = {}
        needs_escalation = []

        for rate in self.rates:
            target_words = condense.target_word_count(self.text, rate)
            self._target_words_by_rate[rate] = target_words
            print(f"\n=== {rate}% condensation (target ~{target_words} words) ===")

            result = condense.attempt_condensation_trials(
                self.ordered_sentences, self.cluster_key_terms, target_words, self.corpus_name,
                model=self.ollama_model, max_trials=self.max_trials, include_full_text=False,
            )
            for t in result["trials"]:
                print(f"  trial {t['trial']}: {t['word_count']} words "
                      f"({'OK' if t['within_tolerance'] else 'outside tolerance'})")
                self.decisions2.record(
                    step="condensation_generation", decision_type="trial_result",
                    prompt="Condensation generation trial",
                    extra={"rate": rate, "source": "web", **t},
                )

            # Even when no trial lands within tolerance, keep the closest one
            # (soft_accept) as a fallback rather than discarding it outright --
            # escalation is still offered below, and can overwrite it with a
            # better attempt.
            if result["text"] is not None:
                self.condensed_texts[rate] = result["text"]
            if not result["success"]:
                needs_escalation.append({"rate": rate, "trials": result["trials"]})

        if needs_escalation:
            return {"type": "escalation", "payload": {"rates": needs_escalation}}

        return self._step_injection_analysis()

    def apply_escalation(self, decisions_by_rate):
        """decisions_by_rate: {rate: bool} -- True = retry with full source text."""
        self.decisions2.record(
            step="condensation_generation", decision_type="escalation_batch",
            prompt="Escalation decisions (web batch)",
            choice=f"{len(decisions_by_rate)} rate(s) decided",
            extra={"source": "web", "decisions": decisions_by_rate},
        )
        self._pending_escalation = decisions_by_rate
        self._run_bg(self._step_apply_escalation, "Escalated condensation generation")

    def _step_apply_escalation(self):
        for rate_str, escalate in self._pending_escalation.items():
            rate = int(rate_str)
            if not escalate:
                if rate in self.condensed_texts:
                    print(f"\nKeeping the closest non-escalated trial for {rate}% (not within tolerance).")
                else:
                    print(f"\nGiving up on {rate}% -- no condensation within tolerance.")
                continue

            target_words = self._target_words_by_rate[rate]
            print(f"\n=== {rate}% condensation, escalated (full source text included) ===")
            result = condense.attempt_condensation_trials(
                self.ordered_sentences, self.cluster_key_terms, target_words, self.corpus_name,
                model=self.ollama_model, max_trials=self.max_trials, include_full_text=True,
                source_text=self.text,
            )
            for t in result["trials"]:
                print(f"  [escalated] trial {t['trial']}: {t['word_count']} words "
                      f"({'OK' if t['within_tolerance'] else 'outside tolerance'})")
                self.decisions2.record(
                    step="condensation_generation", decision_type="trial_result",
                    prompt="Condensation generation trial (escalated)",
                    extra={"rate": rate, "escalated": True, "source": "web", **t},
                )

            if result["text"] is not None:
                self.condensed_texts[rate] = result["text"]
                if not result["success"]:
                    print(f"\n{rate}%: no escalated trial hit tolerance -- using the closest one instead of discarding it.")
            else:
                print(f"\nGiving up on {rate}% after escalation -- no condensation within tolerance.")

        return self._step_injection_analysis()

    # ------------------------------------------------------------
    # PHASE 2: injection analysis -> borderline review -> reports
    # ------------------------------------------------------------

    def _step_injection_analysis(self):
        for rate, condensed_text in self.condensed_texts.items():
            condensed_path = self.paths.condensation_paths(rate)["condensed_text"]
            with open(condensed_path, "w", encoding="utf-8") as f:
                f.write(condensed_text)
            print(f"Saved condensed text to {condensed_path}")

        self.injection_results = {}
        for rate, condensed_text in self.condensed_texts.items():
            print(f"\n=== Injection analysis: {rate}% condensation ===")
            all_spans, source_sentences = condense.classify_condensation(condensed_text, self.text, self.nlp)
            stats = condense.compute_injection_stats(all_spans, condensed_text, self.text)
            borderline_flags = condense.flag_borderline_classifications(all_spans)
            print(f"Spans classified: {len(all_spans)}; borderline flags: {len(borderline_flags)}")

            self.injection_results[rate] = {
                "all_spans": all_spans, "source_sentences": source_sentences,
                "stats": stats, "borderline_flags": borderline_flags,
            }

        flags_payload = [
            {"rate": rate, "flags": result["borderline_flags"]}
            for rate, result in self.injection_results.items() if result["borderline_flags"]
        ]
        if flags_payload:
            return {"type": "injection_review", "payload": {"rates": flags_payload}}

        return self._step_build_reports()

    def apply_injection_review(self, reclassifications_by_rate):
        """reclassifications_by_rate: {rate: {span_id: new_type_or_"keep"}}."""
        self.decisions2.record(
            step="injection_review", decision_type="borderline_reclassification_batch",
            prompt="Borderline reclassification (web batch)",
            choice=f"{len(reclassifications_by_rate)} rate(s) reviewed",
            extra={"source": "web", "decisions": reclassifications_by_rate},
        )

        for rate_str, span_decisions in reclassifications_by_rate.items():
            rate = int(rate_str)
            spans_by_id = {s["span_id"]: s for s in self.injection_results[rate]["all_spans"]}
            for span_id, new_type in span_decisions.items():
                if new_type and new_type != "keep" and span_id in spans_by_id:
                    spans_by_id[span_id]["type"] = new_type

        # A regeneration in flight only ever reviews its own single rate --
        # finish just that rate instead of rebuilding every rate's reports.
        if self._regen_rate is not None:
            self._run_bg(self._step_finish_regeneration, "Building reports")
        else:
            self._run_bg(self._step_build_reports, "Building reports")

    def _process_and_save_rate(self, rate, condensed_text):
        """Runs cluster coverage -> report building -> saving (injection
        report, HTML fragment/preview, human report, plain summary, Voyant
        notebook, standalone report) for one rate, and updates
        self.manifest["rates"][rate] in place. Shared by the initial
        multi-rate build and the post-completion regeneration flow.
        Assumes self.injection_results[rate] is already populated, and
        that self.manifest already exists (it does by the time this is
        ever called -- either just-created by the caller, or from the
        first completed build, for regeneration)."""
        result = self.injection_results[rate]
        coverage = condense.compute_cluster_coverage(self.phase1_state, condensed_text, self.stemmer)
        self.coverage_results[rate] = coverage

        fragment = condensation_report.build_condensation_fragment(
            condensed_text, result["all_spans"], result["source_sentences"], coverage,
            corpus_name=self.corpus_name, rate=rate,
            source_word_count=condense.count_words(self.text),
            phase1_json_name=self.paths.phase1_state_json().name,
        )
        self.fragments_by_rate[rate] = fragment

        preview = condensation_report.build_standalone_preview(fragment, self.corpus_name)
        human_report = condensation_report.build_human_report(
            result["all_spans"], result["borderline_flags"], coverage,
            result["stats"]["verbatim_overlap_pct"], result["stats"]["non_injected_pct"],
        )
        blocks = condensation_report.parse_condensed_blocks(condensed_text)
        plain_summary = condensation_report.build_plain_summary(blocks)

        injection_report_data = {
            "corpus": self.corpus_name, "rate": rate,
            "spans": result["all_spans"], "borderline_flags": result["borderline_flags"],
            "coverage": coverage, "stats": result["stats"],
        }

        output_paths = self.paths.condensation_paths(rate)
        condensation_report.save_condensation_outputs(
            output_paths, fragment, preview, human_report, plain_summary, injection_report_data,
        )

        voyant_html = voyant_notebook.build_voyant_notebook(
            self.enriched_state, {rate: fragment}, self.corpus_name, self.paths,
        )
        voyant_notebook.save_voyant_notebook(voyant_html, self.paths.voyant_notebook_path(rate))

        standalone_html = standalone_report.build_standalone_report(
            self.enriched_state, fragment, self.corpus_name, rate, self.paths,
        )
        standalone_report.save_standalone_report(standalone_html, self.paths.standalone_report_path(rate))

        print(f"\n=== {rate}% condensation outputs written ===")

        self.manifest["rates"][rate] = {k: self._rel(v) for k, v in output_paths.items()}
        self.manifest["rates"][rate]["voyant_notebook"] = self._rel(self.paths.voyant_notebook_path(rate))
        self.manifest["rates"][rate]["standalone_report"] = self._rel(self.paths.standalone_report_path(rate))

        return fragment

    def _step_build_reports(self):
        self.coverage_results = {}
        self.fragments_by_rate = {}
        self.manifest = self._empty_manifest()

        for rate, condensed_text in self.condensed_texts.items():
            self._process_and_save_rate(rate, condensed_text)

        print("\nPipeline complete.")
        return None

    # ------------------------------------------------------------
    # PHASE 2: post-completion condensation regeneration
    # ------------------------------------------------------------

    def preview_regeneration_prompt(self, rate):
        """Renders (but doesn't send) the default condensation prompt for
        the given rate, from this session's already-computed informative
        sentences/cluster terms -- lets the web UI show/edit it before
        actually triggering a regeneration."""
        if getattr(self, "ordered_sentences", None) is None:
            raise RuntimeError("No condensation was set up for this run -- nothing to regenerate.")
        target_words = condense.target_word_count(self.text, rate)
        return condense.build_condensation_prompt(
            self.ordered_sentences, self.cluster_key_terms, target_words, self.corpus_name,
        )

    def start_regeneration(self, rate, prompt_override):
        """Regenerates one condensation rate: optionally at a new target
        rate (the caller just passes whatever rate it wants -- an existing
        one is overwritten in place, a new one is added alongside the
        rest), optionally with a fully researcher-edited prompt
        (prompt_override, or None/empty to use the freshly rendered
        default). Only callable once the pipeline has reached "complete"
        -- reuses the same background-job/polling machinery every other
        step already uses, so the frontend just watches status flip back
        to "complete" with an updated manifest."""
        if self.status != "complete":
            raise RuntimeError("Can only regenerate a condensation once the pipeline has completed.")
        if getattr(self, "ordered_sentences", None) is None:
            raise RuntimeError("No condensation was set up for this run -- nothing to regenerate.")

        self.decisions2.record(
            step="condensation_regeneration", decision_type="regenerate",
            prompt="Regenerate a condensation", choice=f"rate={rate}",
            extra={"source": "web", "rate": rate, "prompt_overridden": bool(prompt_override)},
        )

        self._regen_rate = rate
        self._regen_prompt_override = prompt_override or None
        self._run_bg(self._step_regenerate_generate, f"Regenerating {rate}% condensation")

    def _step_regenerate_generate(self):
        rate = self._regen_rate
        target_words = condense.target_word_count(self.text, rate)

        print(f"\n=== Regenerating {rate}% condensation (target ~{target_words} words) ===")
        result = condense.attempt_condensation_trials(
            self.ordered_sentences, self.cluster_key_terms, target_words, self.corpus_name,
            model=self.ollama_model, max_trials=self.max_trials, include_full_text=False,
            prompt_override=self._regen_prompt_override,
        )
        for t in result["trials"]:
            print(f"  trial {t['trial']}: {t['word_count']} words "
                  f"({'OK' if t['within_tolerance'] else 'outside tolerance'})")
            self.decisions2.record(
                step="condensation_regeneration", decision_type="trial_result",
                prompt="Condensation regeneration trial",
                extra={"rate": rate, "source": "web", **t},
            )

        if result["text"] is None:
            print(f"\nRegeneration at {rate}% failed -- no output produced. "
                  "Previous outputs for this rate, if any, are unchanged.")
            self._regen_rate = None
            return None
        if not result["success"]:
            note = "using the closest trial instead of discarding it" if result["soft_accept"] else "no valid trial"
            print(f"\n{rate}%: no trial hit the target word count within tolerance -- {note}.")

        condensed_text = result["text"]
        self.condensed_texts[rate] = condensed_text

        condensed_path = self.paths.condensation_paths(rate)["condensed_text"]
        with open(condensed_path, "w", encoding="utf-8") as f:
            f.write(condensed_text)
        print(f"Saved condensed text to {condensed_path}")

        print(f"\n=== Injection analysis: {rate}% condensation ===")
        all_spans, source_sentences = condense.classify_condensation(condensed_text, self.text, self.nlp)
        stats = condense.compute_injection_stats(all_spans, condensed_text, self.text)
        borderline_flags = condense.flag_borderline_classifications(all_spans)
        print(f"Spans classified: {len(all_spans)}; borderline flags: {len(borderline_flags)}")

        self.injection_results[rate] = {
            "all_spans": all_spans, "source_sentences": source_sentences,
            "stats": stats, "borderline_flags": borderline_flags,
        }

        if borderline_flags:
            return {"type": "injection_review", "payload": {"rates": [{"rate": rate, "flags": borderline_flags}]}}

        return self._step_finish_regeneration()

    def _step_finish_regeneration(self):
        rate = self._regen_rate
        self._process_and_save_rate(rate, self.condensed_texts[rate])
        self._regen_rate = None
        print("\nPipeline complete.")
        return None
