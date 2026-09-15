"""PipelineSession: the web UI's orchestration layer.

This is a fourth way to drive the same pipeline the notebooks and
run_pipeline.py already run -- it calls the exact same functions in
phase1/pipeline.py, phase2/pipeline.py, phase2/condense.py,
phase2/condensation_report.py, phase3/pipeline.py,
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

from common.paths import CorpusPaths
from common.decisions import DecisionLog
from common import standalone_report, file_convert, parameter_sweep, resume
from phase0.clean_corpus import clean_text
from phase1 import pipeline as phase1_pipeline
from phase2 import pipeline as phase2_pipeline, condense, condensation_report
from phase3 import pipeline as phase3_pipeline, collocations as phase3_collocations, report as phase3_report


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
        self.doc = None
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
        """config: dict with frequency_percentile, max_stems, max_sentences_per_stem,
        max_synsets, max_cluster_size, min_clusters, min_cluster_len,
        glossbert_model, lang_model, sentence_embedder (already typed by app.py)."""
        self.corpus_name = corpus_name
        self.paths = CorpusPaths(corpus_name)
        self.paths.ensure_dirs()
        self.decisions1 = DecisionLog(corpus_name, phase="phase1")
        self.decisions2 = DecisionLog(corpus_name, phase="phase2")
        self.config = config

        # .txt/.md/.markdown/.pdf all funnel through the same converter --
        # whatever format was uploaded, raw/<corpus>.txt always ends up
        # holding plain text. Errors from here (unsupported extension,
        # non-UTF-8 .txt/.md, an unextractable PDF) propagate to the
        # caller (app.py's /api/start), same as before this format
        # support existed.
        raw_text = file_convert.convert_to_text(file_storage.read(), file_storage.filename)
        # Kept verbatim (pre-Phase-0) specifically for source-metadata
        # detection below -- title/author/date front matter sits at the
        # very start of the document, and Phase 0 cleaning isn't
        # guaranteed to leave it alone (e.g. a title that also repeats as
        # a running header gets stripped as a recurring short line), so
        # detection always reads from this untouched copy rather than
        # self.text, whether or not cleaning was requested.
        self._raw_source_text = raw_text
        with open(self.paths.raw_txt(), "w", encoding="utf-8") as f:
            f.write(clean_text(raw_text) if clean else raw_text)

        if self.config.get("run_parameter_sweep"):
            self._run_bg(self._step_run_parameter_sweep, "Sweeping frequency_percentile candidates")
        else:
            self._run_bg(self._step_start_to_flagged_review, "Extracting stems & running GlossBERT analysis")

    def detect_metadata(self, ollama_model):
        """Runs extract_source_metadata against the untouched pre-Phase-0
        source text captured in start(), using the given Ollama model.
        Called by the web UI right as the Phase 2 setup screen loads, so
        the title/author/date fields open already pre-filled with the
        model's best guess (or "Unclear") for the researcher to confirm
        or edit, rather than asking them to type it in blind.

        Printed to the server's own console (real stdout, not the
        session log the browser polls -- this runs in the Flask request
        thread, outside _run_bg's tee) since this call has no other
        visible trail if it silently fails client-side: the researcher
        only sees a small status line on the form, easy to miss, and the
        browser's log panel never receives this route's output at all."""
        print(f"[detect-metadata] request: model={ollama_model!r}")
        if getattr(self, "_raw_source_text", None) is None:
            print("[detect-metadata] no corpus loaded yet")
            raise RuntimeError("No corpus loaded yet.")
        title, authors, date = condense.extract_source_metadata(self._raw_source_text, ollama_model)
        print(f"[detect-metadata] result: title={title!r}, authors={authors!r}, date={date!r}")
        return {"title": title, "authors": authors, "date": date}

    # ------------------------------------------------------------
    # ALTERNATIVE ENTRY POINT: resume an existing corpus at Phase 2 or 3
    # ------------------------------------------------------------

    def resume(self, corpus_name, start_phase: int, config: dict | None = None):
        """Alternative to start(): resumes an already-existing corpus at
        Phase 2 or 3 instead of uploading a new file and running from
        Phase 1. Shares prerequisite-checking with the CLI's
        --start-phase via common/resume.py -- a corpus missing a required
        file from an earlier phase is rejected here with the exact same
        check the CLI uses, not a second, possibly-inconsistent one.
        Raises RuntimeError (caught by app.py's route) if prerequisites
        aren't met; never partially starts."""
        if start_phase not in (2, 3):
            raise RuntimeError(f"resume() only supports start_phase 2 or 3, got {start_phase}")

        self.corpus_name = corpus_name
        self.paths = CorpusPaths(corpus_name)
        self.config = config or {}

        # Checked BEFORE creating anything else -- DecisionLog's own
        # __init__ creates data/<corpus>/decisions/ on construction, and
        # ensure_dirs() creates the rest, so both must wait until AFTER
        # this check passes. Otherwise a failed check for a
        # nonexistent/incomplete corpus would still leave behind a full
        # tree of empty data/<corpus>/* directories (same fix as the
        # CLI's --start-phase; see run_pipeline.py::main -- verified this
        # exact leak happening here too before this ordering fix).
        ok, missing, available_rates = resume.check_prerequisites(self.paths, start_phase)
        if not ok:
            raise RuntimeError(
                f"Cannot resume corpus {corpus_name!r} at Phase {start_phase}: missing "
                + "; ".join(missing)
            )

        self.paths.ensure_dirs()
        self.decisions1 = DecisionLog(corpus_name, phase="phase1")
        self.decisions2 = DecisionLog(corpus_name, phase="phase2")

        if start_phase == 2:
            self._run_bg(self._step_resume_at_phase2, "Resuming at Phase 2")
        else:
            self._resume_available_rates = available_rates
            self._run_bg(self._step_resume_at_phase3, "Resuming at Phase 3")

    def _resolve_lang_model(self):
        """Recovers which spaCy model Phase 1 actually used from its own
        decision log (see phase1_pipeline.log_phase1_config) -- older
        runs that predate that logging fall back to the resume form's own
        lang_model field, then the same default every interface uses."""
        return (
            resume.find_last_decision_choice(self.corpus_name, "phase1", "lang_model_choice")
            or self.config.get("lang_model", "en_core_web_sm")
        )

    def _step_resume_at_phase2(self):
        """Loads everything a normal run's _step_finish_phase1 would
        already have produced, from disk instead -- then returns the
        exact same "phase2_setup" decision-gate shape, so the existing
        Phase 2 setup screen and everything downstream of it
        (apply_phase2_setup onward) runs completely unmodified."""
        self.phase1_state = phase2_pipeline.load_phase1_state(self.paths.phase1_state_json())
        with open(self.paths.raw_txt(), "r", encoding="utf-8") as f:
            self.text = f.read()
        # The true pre-Phase-0-clean original text (see start()) only
        # ever existed in memory during the original run, never
        # persisted -- source-metadata detection falls back to the saved
        # raw text instead, which may already be Phase-0-cleaned.
        self._raw_source_text = self.text
        self.nlp = spacy.load(self._resolve_lang_model())
        self.stemmer = PorterStemmer()
        self.doc = None

        print(f"Resumed corpus {self.corpus_name!r} at Phase 2 (loaded {self.paths.phase1_state_json()}).")
        return {
            "type": "phase2_setup",
            "payload": {
                "cluster_count": len(self.phase1_state.get("clusterDefs", [])),
                "phase1_state_path": str(self.paths.phase1_state_json()),
                "resumed": True,
            },
        }

    def _step_resume_at_phase3(self):
        """Loads everything a normal run would already have produced by
        the time Phase 3 starts (Phase 1 state, corpus text, and every
        already-generated condensed rate discovered on disk), rebuilds
        the manifest those rates' own outputs already populate (so the
        completion screen shows them, not just the freshly-rebuilt Phase
        3 report), then hands off to the exact same _step_prepare_phase3
        a normal run uses -- collocation-pair selection (if applicable)
        and the distant-reading report build are completely unmodified."""
        self.phase1_state = phase2_pipeline.load_phase1_state(self.paths.phase1_state_json())
        with open(self.paths.raw_txt(), "r", encoding="utf-8") as f:
            self.text = f.read()
        self.nlp = spacy.load(self._resolve_lang_model())
        self.stemmer = PorterStemmer()

        self.condensed_texts = {}
        for rate in self._resume_available_rates:
            with open(self.paths.condensation_paths(rate)["condensed_text"], "r", encoding="utf-8") as f:
                self.condensed_texts[rate] = f.read()

        self.manifest = self._empty_manifest()
        for rate in self._resume_available_rates:
            output_paths = self.paths.condensation_paths(rate)
            self.manifest["rates"][rate] = {k: self._rel(v) for k, v in output_paths.items()}
            self.manifest["rates"][rate]["standalone_report"] = self._rel(self.paths.standalone_report_path(rate))

        print(f"Resumed corpus {self.corpus_name!r} at Phase 3 "
              f"(reusing rate(s): {self._resume_available_rates}).")
        return self._step_prepare_phase3()

    # ------------------------------------------------------------
    # PHASE 1: optional parameter sweep -> stems -> GlossBERT -> flagged-term review
    # ------------------------------------------------------------

    def _step_run_parameter_sweep(self):
        """Runs before stem extraction, only when the researcher opted into
        the "run_parameter_sweep" checkbox. Sets up nlp/stemmer/text/doc
        here (rather than only in _step_start_to_flagged_review) so the
        sweep and the eventual real extraction share one spaCy parse --
        see that step's `if self.doc is None` guard."""
        self.nlp = spacy.load(self.config["lang_model"])
        self.stemmer = PorterStemmer()

        with open(self.paths.raw_txt(), "r", encoding="utf-8") as f:
            self.text = f.read()
        self.doc = self.nlp(self.text)

        word_frequency_percentile = self.config.get("word_frequency_percentile", 0.0)
        rows = parameter_sweep.compute_sweep_report(
            self.doc, self.text, self.nlp, self.stemmer,
            word_frequency_percentile=word_frequency_percentile,
        )
        self.decisions1.record(
            step="phase1_parameter_sweep", decision_type="sweep_shown",
            prompt="Frequency-percentile sweep candidates", choice=None,
            extra={
                "candidates": rows, "max_stems": self.config["max_stems"],
                "word_frequency_percentile": word_frequency_percentile, "source": "web",
            },
        )
        return {
            "type": "parameter_sweep",
            "payload": {"candidates": rows, "max_stems": self.config["max_stems"]},
        }

    def apply_parameter_sweep(self, choice, frequency_percentile, max_stems):
        self.decisions1.record(
            step="phase1_parameter_sweep", decision_type="frequency_percentile_choice",
            prompt="Pick a candidate, or provide a custom frequency_percentile/max_stems",
            choice=choice,
            extra={
                "resolved_frequency_percentile": frequency_percentile, "resolved_max_stems": max_stems,
                "custom": choice == "custom", "source": "web",
            },
        )
        self.config["frequency_percentile"] = frequency_percentile
        self.config["max_stems"] = max_stems
        self._run_bg(self._step_start_to_flagged_review, "Extracting stems & running GlossBERT analysis")

    def _step_start_to_flagged_review(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        if self.doc is None:
            self.nlp = spacy.load(self.config["lang_model"])
            self.stemmer = PorterStemmer()
            with open(self.paths.raw_txt(), "r", encoding="utf-8") as f:
                self.text = f.read()
            self.doc = self.nlp(self.text)

        self.top_stems = phase1_pipeline.extract_top_stems(
            self.doc, self.stemmer, self.config["frequency_percentile"], self.config["max_stems"]
        )
        cap_note = phase1_pipeline.describe_stem_cap(
            self.top_stems, self.doc, self.stemmer, self.config["frequency_percentile"], self.config["max_stems"]
        )
        print(f"Selected {len(self.top_stems)} top stems" + (f" {cap_note}" if cap_note else ""))

        with open(self.paths.top_stems(), "w", encoding="utf-8") as out:
            out.write("stem\tcount\n")
            for stem, count in self.top_stems.items():
                out.write(f"{stem}\t{count}\n")
        print(f"Saved results to {self.paths.top_stems()}")

        print("Loading GlossBERT checkpoint...")
        self.tokenizer, self.model = phase1_pipeline.load_glossbert(self.config["glossbert_model"], self.device)
        print("GlossBERT loaded successfully.")

        self.stem_occurrences = phase1_pipeline.map_stems_to_sentences(self.doc, self.top_stems, self.stemmer)
        word_frequency_percentile = self.config.get("word_frequency_percentile", 0.0)
        if word_frequency_percentile > 0:
            self.stem_occurrences = phase1_pipeline.filter_stem_occurrences_by_word_frequency(
                self.stem_occurrences, word_frequency_percentile,
            )
            print(f"Filtered derived words to word_frequency_percentile={word_frequency_percentile}")

        phase1_pipeline.log_phase1_config(self.decisions1, {
            "frequency_percentile": self.config["frequency_percentile"],
            "max_stems": self.config["max_stems"],
            "word_frequency_percentile": word_frequency_percentile,
            "max_sentences_per_stem": self.config["max_sentences_per_stem"],
            "max_synsets": self.config["max_synsets"],
            "merge_duplicate_word_occurrences": self.config["merge_duplicate_word_occurrences"],
            "max_cluster_size": self.config["max_cluster_size"],
            "min_clusters": self.config["min_clusters"],
            "min_cluster_len": self.config["min_cluster_len"],
            "glossbert_model": self.config["glossbert_model"],
            "lang_model": self.config["lang_model"],
            "sentence_embedder": self.config["sentence_embedder"],
        }, source="web")

        print("Running GlossBERT analysis...")
        self.accepted_definitions, self.flagged_words = phase1_pipeline.run_glossbert_analysis(
            self.top_stems, self.stem_occurrences, self.tokenizer, self.model, self.device,
            max_sentences_per_stem=self.config["max_sentences_per_stem"], max_synsets=self.config["max_synsets"],
            merge_duplicate_word_occurrences=self.config["merge_duplicate_word_occurrences"],
        )
        print(f"Done. {len(self.flagged_words)} flagged term(s) need review.")
        excluded_note = phase1_pipeline.describe_wordnet_excluded_stems(
            self.stem_occurrences, max_sentences_per_stem=self.config["max_sentences_per_stem"],
            merge_duplicate_word_occurrences=self.config["merge_duplicate_word_occurrences"],
        )
        if excluded_note:
            print(excluded_note)

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
            elif choice == "delete":
                phase1_pipeline.purge_word_from_occurrences(self.stem_occurrences, item["stem"], item["word"])
                continue
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
        self.clusters, large_subclusters, reembedded_main = phase1_pipeline.recluster_large_clusters(
            self.clusters, self.stem_occurrences, self.embedder, self.tokenizer, self.model, self.device,
            max_cluster_size=self.config["max_cluster_size"], min_clusters=self.config["min_clusters"],
            min_cluster_len=self.config["min_cluster_len"], max_synsets=self.config["max_synsets"],
        )
        print(f"{len(large_subclusters)} large-cluster subcluster(s) found.")
        still_oversized = phase1_pipeline.find_oversized_clusters(self.clusters, self.config["max_cluster_size"])
        if still_oversized:
            print(f"WARNING: {len(still_oversized)} cluster(s) are still over max_cluster_size "
                  f"({self.config['max_cluster_size']}) after splitting -- sizes: "
                  f"{[len(set(v)) for v in still_oversized.values()]}. The data didn't separate further; "
                  "review these in the cluster-review step below.")

        noise_stems = phase1_pipeline.extract_noise_stems(self.stem_names, self.labels)
        self.noise_clusters = phase1_pipeline.recluster_noise(
            noise_stems, self.stem_occurrences, self.embedder, self.tokenizer, self.model, self.device,
            min_clusters=self.config["min_clusters"], min_cluster_len=self.config["min_cluster_len"],
            max_synsets=self.config["max_synsets"],
        )
        print(f"{len(self.noise_clusters)} noise cluster(s) recovered.")

        # recluster_noise's own HDBSCAN pass can just as easily produce an
        # oversized cluster as the primary pass does (its default
        # cluster-selection method favors one large stable cluster over
        # many small ones) -- run its output through the same
        # max_cluster_size split the primary pass's clusters already got
        # above, rather than letting it skip that safety net purely
        # because of which pass produced it.
        reembedded_noise_direct = {s for stems in self.noise_clusters.values() for s in stems}
        self.noise_clusters, noise_large_subclusters, reembedded_noise_resplit = phase1_pipeline.recluster_large_clusters(
            self.noise_clusters, self.stem_occurrences, self.embedder, self.tokenizer, self.model, self.device,
            max_cluster_size=self.config["max_cluster_size"], min_clusters=self.config["min_clusters"],
            min_cluster_len=self.config["min_cluster_len"], max_synsets=self.config["max_synsets"],
        )
        if noise_large_subclusters:
            print(f"{len(noise_large_subclusters)} oversized noise-recovered subcluster(s) split out.")
        still_oversized = phase1_pipeline.find_oversized_clusters(self.noise_clusters, self.config["max_cluster_size"])
        if still_oversized:
            print(f"WARNING: {len(still_oversized)} noise-recovered cluster(s) are still over "
                  f"max_cluster_size ({self.config['max_cluster_size']}) after splitting -- sizes: "
                  f"{[len(set(v)) for v in still_oversized.values()]}. The data didn't separate further; "
                  "review these in the cluster-review step below.")

        # Stems whose cluster placement came from _split_oversized_clusters_once
        # or recluster_noise: their clustering embedding used GlossBERT's fresh
        # top candidate senses, not the researcher's accepted definition, since
        # neither reclustering function ever sees accepted_definitions.
        reembedded_noise = reembedded_noise_direct | reembedded_noise_resplit

        stem_word_frequencies = phase1_pipeline.build_stem_word_frequency_table(self.stem_occurrences)
        renamed_clusters = phase1_pipeline.rename_clusters(self.clusters, stem_word_frequencies, reembedded_main, reembedded_source="main")
        renamed_noise_clusters = phase1_pipeline.rename_clusters(self.noise_clusters, stem_word_frequencies, reembedded_noise, reembedded_source="noise")
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
                    {
                        "name": name, "stems": data["stems"], "ngrams": data.get("ngrams", []),
                        "reembedded_stems": data.get("reembedded_stems", []),
                        "reembedded_source": data.get("reembedded_source"),
                    }
                    for name, data in sorted(self.renamed_clusters.items())
                ]
            },
        }

    def apply_cluster_review(self, entries):
        """entries: list of {original_name, removed?, new_name, kept_stems,
        removed_stems, kept_ngrams, removed_ngrams}. An entry with
        removed=true is dropped entirely -- it never appears in
        final_clusters (mirrors the CLI's "d = drop entirely" option)."""
        self.decisions1.record(
            step="cluster_review", decision_type="batch_review",
            prompt="Cluster review (web batch)",
            choice=f"{len(entries)} cluster(s) reviewed",
            extra={"source": "web", "decisions": entries},
        )

        final_clusters = {}
        excluded_cluster_ngrams = set()

        for entry in entries:
            if entry.get("removed"):
                continue

            new_name = (entry.get("new_name") or "").strip() or entry["original_name"]
            excluded_cluster_ngrams.update(entry.get("removed_ngrams", []))

            # reembedded_stems/reembedded_source aren't something the researcher
            # edits, so they're looked up server-side from the original cluster
            # data (already computed in _step_after_flagged_review) rather than
            # round-tripped through the browser -- filtered to stems that
            # survived review, same as the CLI's run_cluster_review does.
            original_cluster = self.renamed_clusters.get(entry["original_name"], {})
            original_reembedded = original_cluster.get("reembedded_stems", [])
            kept_stems = entry["kept_stems"]

            final_clusters[new_name] = {
                "stems": kept_stems,
                "ngrams": entry.get("kept_ngrams", []),
                "excluded_stems": entry.get("removed_stems", []),
                "excluded_ngrams": entry.get("removed_ngrams", []),
                "reembedded_stems": [s for s in original_reembedded if s in kept_stems],
                "reembedded_source": original_cluster.get("reembedded_source"),
            }

        self.final_clusters = final_clusters
        self.excluded_cluster_ngrams = excluded_cluster_ngrams

        self._run_bg(self._step_finish_phase1, "Finishing Phase 1")

    # ------------------------------------------------------------
    # PHASE 1: finish & save
    # ------------------------------------------------------------

    def _step_finish_phase1(self):
        self.phase1_state = phase1_pipeline.build_phase1_state(
            self.final_clusters, self.excluded_cluster_ngrams,
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
                            rates_str, ollama_model, max_trials, adversarial_model=None,
                            title="", authors="", date=""):
        self.decisions2.record(
            step="phase2_setup", decision_type="sentence_selection_config",
            prompt="Phase 2 sentence-selection config",
            choice=f"top_n={top_n}, density_percentile={density_percentile}",
            extra={"source": "web"},
        )
        self.decisions2.record(
            step="phase2_setup", decision_type="run_condensation_choice",
            prompt="Generate a condensation?",
            choice=str(run_condensation), extra={"source": "web"},
        )

        self.top_n = top_n
        self.density_percentile = density_percentile
        self.run_condensation = run_condensation
        self.rates = []
        # title/authors/date: the browser already ran detect_metadata (an
        # AI guess against the untouched pre-Phase-0 text, see start())
        # and pre-filled the form with it before the researcher saw this
        # screen -- whatever comes back here, edited or not, is the
        # researcher's confirmed answer, so it's taken as final.
        self.title = title.strip() or "Unclear"
        self.authors = authors.strip() or "Unclear"
        self.date = date.strip() or "Unclear"

        if run_condensation:
            self.rates = [int(r.strip()) for r in rates_str.split(",") if r.strip()]
            self.ollama_model = ollama_model
            self.adversarial_model = (adversarial_model or "").strip() or ollama_model
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
                step="condensation_setup", decision_type="adversarial_model_choice",
                prompt="Which Ollama model should the adversarial reviewer use?",
                choice=self.adversarial_model, extra={"source": "web"},
            )
            self.decisions2.record(
                step="condensation_setup", decision_type="max_trials_choice",
                prompt="How many generation trials before asking to escalate?",
                choice=str(max_trials), extra={"source": "web"},
            )
            self.decisions2.record(
                step="source_metadata", decision_type="metadata_result",
                prompt="Source title/author(s)/date, confirmed by researcher",
                choice="confirmed",
                extra={"source": "web", "title": self.title, "authors": self.authors, "date": self.date},
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
            self.condensed_texts = {}
            self.manifest = self._empty_manifest()
            print("No condensation requested.")
            return self._step_prepare_phase3()

        print(f"\nSource metadata (researcher-confirmed) -- Title: {self.title}; Author(s): {self.authors}; Date: {self.date}")

        self.ordered_sentences = condense.gather_ordered_informative_sentences(self.enriched_state)
        self.cluster_key_terms = condense.gather_cluster_key_terms(self.enriched_state)

        self.condensed_texts = {}
        self._target_words_by_rate = {}
        self._escalation_sanity_failed_by_rate = {}
        # Trials that hit the word-count target but failed a sanity check
        # are never auto-picked (see attempt_condensation_trials) -- they
        # go to a researcher review step instead. These two dicts track
        # what's pending review and which context it came from ("initial"
        # -> declining offers escalation; "escalated" or "regen" ->
        # declining just gives up, no further auto-retry).
        self._sanity_review_pending = {}
        self._sanity_review_context = {}
        needs_sanity_review = []
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
                print(condense.format_trial_line(t))
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

            if result["sanity_review_candidates"]:
                self._sanity_review_pending[rate] = result
                self._sanity_review_context[rate] = "initial"
                needs_sanity_review.append({"rate": rate, "candidates": self._sanity_candidates_payload(result)})
            elif not result["success"]:
                # Kept so a later "decline escalation" can report the real
                # reason -- a trial can fail here either for missing the
                # word-count target or for failing the sanity check (e.g.
                # a repeated-phrase degeneration) even while landing
                # within tolerance, and those are different situations.
                self._escalation_sanity_failed_by_rate[rate] = bool(result.get("sanity_failed"))
                needs_escalation.append({"rate": rate, "trials": result["trials"]})

        # Sanity review always comes first -- a rate rejected there can
        # still fall through to escalation afterward (see
        # _step_apply_sanity_review), so escalation for the rest is
        # deferred rather than decided here.
        self._deferred_escalation = needs_escalation

        if needs_sanity_review:
            return {"type": "sanity_review", "payload": {"rates": needs_sanity_review}}

        if needs_escalation:
            return {"type": "escalation", "payload": {"rates": needs_escalation}}

        return self._step_injection_analysis()

    def _sanity_candidates_payload(self, result):
        """Trial number/word-count/target/sanity-issues for each
        tolerance-hit-but-sanity-failed candidate -- no trial text, same
        information the CLI's run_sanity_review_for_rate prints. The
        actual text stays server-side in self._sanity_review_pending
        until the researcher picks a trial number."""
        return [
            {"trial": c["trial"], "word_count": c["word_count"],
             "target_words": c["target_words"], "sanity_issues": c["sanity_issues"]}
            for c in result["sanity_review_candidates"]
        ]

    def apply_sanity_review(self, decisions_by_rate):
        """decisions_by_rate: {rate: trial_number|None} -- which
        sanity-flagged trial (if any) to keep for each rate under
        review."""
        self.decisions2.record(
            step="condensation_generation", decision_type="sanity_review_batch",
            prompt="Sanity-check review decisions (web batch)",
            choice=f"{len(decisions_by_rate)} rate(s) decided",
            extra={"source": "web", "decisions": decisions_by_rate},
        )
        self._pending_sanity_review = decisions_by_rate
        self._run_bg(self._step_apply_sanity_review, "Applying sanity-check review decisions")

    def _step_apply_sanity_review(self):
        needs_escalation = list(getattr(self, "_deferred_escalation", []))
        regen_accepted_text = None

        for rate_str, trial_choice in self._pending_sanity_review.items():
            rate = int(rate_str)
            result = self._sanity_review_pending.pop(rate, None)
            context = self._sanity_review_context.pop(rate, "initial")
            if result is None:
                continue

            candidates = result["sanity_review_candidates"]
            accepted = None
            if trial_choice is not None:
                accepted = next((c for c in candidates if c["trial"] == int(trial_choice)), None)

            if accepted is not None:
                print(f"\nAccepted trial {accepted['trial']} for {rate}% despite its sanity flag.")
                if context == "regen":
                    regen_accepted_text = accepted["text"]
                else:
                    self.condensed_texts[rate] = accepted["text"]
            elif context == "initial":
                self.condensed_texts.pop(rate, None)
                self._escalation_sanity_failed_by_rate[rate] = False
                needs_escalation.append({"rate": rate, "trials": result["trials"]})
                print(f"\nRejected all sanity-flagged trials for {rate}% -- offering escalation.")
            elif context == "escalated":
                self.condensed_texts.pop(rate, None)
                print(f"\nGiving up on {rate}% -- no condensation accepted after escalation.")
            else:  # "regen"
                print(f"\nRegeneration at {rate}% rejected -- previous outputs for this rate, if any, are unchanged.")

        self._deferred_escalation = []

        if needs_escalation:
            return {"type": "escalation", "payload": {"rates": needs_escalation}}

        if getattr(self, "_regen_rate", None) is not None:
            if regen_accepted_text is not None:
                return self._step_regenerate_after_text(self._regen_rate, regen_accepted_text)
            self._regen_rate = None
            return None

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
        needs_sanity_review = []

        for rate_str, escalate in self._pending_escalation.items():
            rate = int(rate_str)
            if not escalate:
                if rate in self.condensed_texts:
                    if self._escalation_sanity_failed_by_rate.get(rate):
                        print(f"\nKeeping the closest non-escalated trial for {rate}% "
                              "(it failed a generation sanity check, e.g. a repeated-phrase "
                              "degeneration -- see the trial log above for which one; not "
                              "necessarily a word-count problem).")
                    else:
                        print(f"\nKeeping the closest non-escalated trial for {rate}% "
                              "(none hit the target word count within tolerance).")
                else:
                    print(f"\nGiving up on {rate}% -- no usable trial produced.")
                continue

            target_words = self._target_words_by_rate[rate]
            print(f"\n=== {rate}% condensation, escalated (full source text included) ===")
            result = condense.attempt_condensation_trials(
                self.ordered_sentences, self.cluster_key_terms, target_words, self.corpus_name,
                model=self.ollama_model, max_trials=self.max_trials, include_full_text=True,
                source_text=self.text,
            )
            for t in result["trials"]:
                print(condense.format_trial_line(t, prefix="[escalated] "))
                self.decisions2.record(
                    step="condensation_generation", decision_type="trial_result",
                    prompt="Condensation generation trial (escalated)",
                    extra={"rate": rate, "escalated": True, "source": "web", **t},
                )

            if result["sanity_review_candidates"]:
                self._sanity_review_pending[rate] = result
                self._sanity_review_context[rate] = "escalated"
                needs_sanity_review.append({"rate": rate, "candidates": self._sanity_candidates_payload(result)})
                continue

            if result["text"] is not None:
                self.condensed_texts[rate] = result["text"]
                if result.get("sanity_failed"):
                    print(f"\n{rate}%: no escalated trial passed the sanity checks -- using the closest one anyway, flag it for a manual look.")
                elif not result["success"]:
                    print(f"\n{rate}%: no escalated trial hit tolerance -- using the closest one instead of discarding it.")
            else:
                print(f"\nGiving up on {rate}% after escalation -- no condensation within tolerance.")

        if needs_sanity_review:
            return {"type": "sanity_review", "payload": {"rates": needs_sanity_review}}

        return self._step_injection_analysis()

    # ------------------------------------------------------------
    # PHASE 2: injection analysis -> borderline review -> reports
    # ------------------------------------------------------------

    def _step_injection_analysis(self):
        self.adversarial_notices = {}
        self._adversarial_issues_by_rate = {}
        for rate in list(self.condensed_texts.keys()):
            condensed_text, was_fixed, _, issues = condense.run_adversarial_review(
                self.condensed_texts[rate], self.ordered_sentences, self._target_words_by_rate[rate],
                self.corpus_name, self.adversarial_model, self.decisions2, rate,
            )
            if was_fixed:
                self.condensed_texts[rate] = condensed_text
                self._adversarial_issues_by_rate[rate] = issues
                self.adversarial_notices[rate] = (
                    f"Adversarial reviewer auto-fixed this condensation ({'; '.join(issues)}). "
                    "The original text is preserved in the Phase 2 decision log."
                )
                print(f"{rate}%: {self.adversarial_notices[rate]}")

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
            {"rate": rate, "flags": self._flags_with_source_context(rate)}
            for rate, result in self.injection_results.items() if result["borderline_flags"]
        ]
        if flags_payload:
            return {"type": "injection_review", "payload": {"rates": flags_payload}}

        return self._step_build_reports()

    def _flags_with_source_context(self, rate):
        """Attaches each borderline flag's matched source sentence(s) --
        resolved from source_refs to actual text -- so the researcher can
        judge a keep/reassign call without keeping the raw source open in
        another window."""
        result = self.injection_results[rate]
        source_sentences = result["source_sentences"]
        enriched = []
        for flag in result["borderline_flags"]:
            flag = dict(flag)
            flag["source_texts"] = [
                source_sentences[idx] for idx in flag.get("source_refs", [])
                if 0 <= idx < len(source_sentences)
            ]
            enriched.append(flag)
        return enriched

    def apply_injection_review(self, reclassifications_by_rate):
        """reclassifications_by_rate: {rate: {span_id: new_type_or_"keep"}}.

        Logs original_type alongside new_type per span (not just the raw
        "keep"/letter choice) -- parity with the CLI/notebook path's
        run_injection_review, which already records both per flag. Without
        this, a posterior check of the webapp's decision log alone can't
        tell what a reclassified span's automatic label actually was
        (recoverable before this only by cross-referencing the saved
        injection report's own borderline_flags array, which happens to
        keep an unmutated copy -- not something a decision-log reader
        should have to know)."""
        logged_decisions = {}
        for rate_str, span_decisions in reclassifications_by_rate.items():
            rate = int(rate_str)
            spans_by_id = {s["span_id"]: s for s in self.injection_results[rate]["all_spans"]}
            logged_decisions[rate_str] = {}
            for span_id, choice in span_decisions.items():
                span = spans_by_id.get(span_id)
                original_type = span["type"] if span is not None else None
                new_type = choice if choice in ("F", "T", "R", "C") else original_type
                logged_decisions[rate_str][span_id] = {
                    "original_type": original_type, "new_type": new_type,
                }
                if choice and choice != "keep" and span is not None:
                    span["type"] = choice

        self.decisions2.record(
            step="injection_review", decision_type="borderline_reclassification_batch",
            prompt="Borderline reclassification (web batch)",
            choice=f"{len(reclassifications_by_rate)} rate(s) reviewed",
            extra={"source": "web", "decisions": logged_decisions},
        )

        # A regeneration in flight only ever reviews its own single rate --
        # finish just that rate instead of rebuilding every rate's reports.
        if self._regen_rate is not None:
            self._run_bg(self._step_finish_regeneration, "Building reports")
        else:
            self._run_bg(self._step_build_reports, "Building reports")

    def _process_and_save_rate(self, rate, condensed_text):
        """Runs cluster coverage -> report building -> saving (injection
        report, HTML fragment/preview, human report, plain summary,
        standalone report) for one rate, and updates
        self.manifest["rates"][rate] in place. Shared by the initial
        multi-rate build and the post-completion regeneration flow.
        Assumes self.injection_results[rate] is already populated, and
        that self.manifest already exists (it does by the time this is
        ever called -- either just-created by the caller, or from the
        first completed build, for regeneration)."""
        result = self.injection_results[rate]
        coverage = condense.compute_cluster_coverage(self.phase1_state, condensed_text, self.text, self.stemmer)
        self.coverage_results[rate] = coverage
        adversarial_notice = getattr(self, "_adversarial_issues_by_rate", {}).get(rate)

        fragment = condensation_report.build_condensation_fragment(
            condensed_text, result["all_spans"], result["source_sentences"], coverage,
            corpus_name=self.corpus_name, rate=rate,
            source_word_count=condense.count_words(self.text),
            phase1_json_name=self.paths.phase1_state_json().name,
            title=self.title, authors=self.authors, date=self.date,
            adversarial_notice=adversarial_notice,
        )
        self.fragments_by_rate[rate] = fragment

        preview = condensation_report.build_standalone_preview(fragment, self.corpus_name)
        sanity_issues = condense.check_condensation_sanity(condensed_text, self.ordered_sentences)
        human_report = condensation_report.build_human_report(
            result["all_spans"], result["borderline_flags"], coverage,
            result["stats"]["verbatim_overlap_pct"], result["stats"]["non_injected_pct"], sanity_issues,
            adversarial_notice=adversarial_notice,
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

        standalone_html = standalone_report.build_standalone_report(
            self.enriched_state, fragment, self.corpus_name, rate, self.paths,
        )
        standalone_report.save_standalone_report(standalone_html, self.paths.standalone_report_path(rate))

        print(f"\n=== {rate}% condensation outputs written ===")

        self.manifest["rates"][rate] = {k: self._rel(v) for k, v in output_paths.items()}
        self.manifest["rates"][rate]["standalone_report"] = self._rel(self.paths.standalone_report_path(rate))
        notice = getattr(self, "adversarial_notices", {}).get(rate)
        if notice:
            self.manifest["rates"][rate]["adversarial_notice"] = notice

        return fragment

    def _step_build_reports(self):
        self.coverage_results = {}
        self.fragments_by_rate = {}
        self.manifest = self._empty_manifest()

        for rate, condensed_text in self.condensed_texts.items():
            self._process_and_save_rate(rate, condensed_text)

        return self._step_prepare_phase3()

    # ------------------------------------------------------------
    # PHASE 3: distant reading + source-vs-summary comparison
    # ------------------------------------------------------------
    # Runs once, automatically, right after Phase 2's initial rates (if
    # any) are built -- a later post-completion regeneration does not
    # retroactively update this report, same as run_pipeline.py's own
    # placement (before its regeneration loop). The only researcher
    # decision here is which collocation pair(s) to compare, and only
    # when there's at least one condensation rate to compare against.

    def _step_prepare_phase3(self):
        print("\n=== Phase 3: distant reading ===")
        self.decisions3 = DecisionLog(self.corpus_name, phase="phase3")
        self.phase3_context = phase3_pipeline.prepare_phase3_context(
            self.phase1_state, self.text, self.nlp, self.stemmer,
        )
        print(
            f"{len(self.phase3_context['stopwords'])} stopword(s) in effect "
            f"({len(self.phase3_context['auto_authors'])} author name(s) auto-detected)."
        )

        if self.condensed_texts and self.phase3_context["colloc_candidates"]:
            self.phase3_skip_note = None
            return {
                "type": "collocation_review",
                "payload": {
                    "candidates": [
                        {
                            "term_a": c["term_a"], "cluster_a": c["cluster_a"],
                            "term_b": c["term_b"], "cluster_b": c["cluster_b"],
                            "hits": c["hits"], "confounded": c["confounded"],
                            "confound_reason": c["confound_reason"],
                        }
                        for c in self.phase3_context["colloc_candidates"]
                    ]
                },
            }

        # No decision to make here -- either surface exactly why (no
        # clusters vs. no real collocations among them), or there's
        # nothing to compare against at all (no condensation), in which
        # case skip_reason is None and there's nothing worth disclosing.
        self.phase3_skip_note = phase3_pipeline.describe_collocation_skip_reason(
            self.phase3_context["clusterdefs"], self.phase3_context["colloc_candidates"], self.condensed_texts,
        )
        if self.phase3_skip_note:
            print(f"Contexts/Collocates comparison skipped: {self.phase3_skip_note}.")
        return self._step_build_phase3_report([])

    def apply_collocation_review(self, selected_indices):
        """selected_indices: list of 0-based ints into
        self.phase3_context['colloc_candidates'] -- the researcher's
        chosen pair(s), resolved the same way as the CLI's
        run_collocation_review but without input()."""
        candidates = self.phase3_context["colloc_candidates"]
        selected_pairs = phase3_collocations.select_pairs_by_index(candidates, selected_indices)

        self.decisions3.record(
            step="collocation_review", decision_type="pair_selection",
            prompt="Which collocation pair(s) to compare between source and summaries?",
            options=[f"{c['term_a']} <-> {c['term_b']}" for c in candidates],
            choice=str(selected_indices),
            extra={"source": "web", "selected": [f"{c['term_a']} <-> {c['term_b']}" for c in selected_pairs]},
        )

        self._run_bg(lambda: self._step_build_phase3_report(selected_pairs), "Building Phase 3 distant-reading report")

    def _step_build_phase3_report(self, selected_pairs):
        html = phase3_pipeline.build_phase3_report(
            self.phase3_context, self.corpus_name, self.phase1_state, self.text, self.stemmer,
            self.condensed_texts, selected_pairs, self.nlp,
        )
        report_path = self.paths.distant_reading_report_path()
        phase3_report.save_distant_reading_report(html, report_path)
        print(f"Distant reading report written to {report_path}")

        self.manifest["distant_reading_report"] = self._rel(report_path)
        if getattr(self, "phase3_skip_note", None):
            self.manifest["distant_reading_note"] = self.phase3_skip_note
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
            print(condense.format_trial_line(t))
            self.decisions2.record(
                step="condensation_regeneration", decision_type="trial_result",
                prompt="Condensation regeneration trial",
                extra={"rate": rate, "source": "web", **t},
            )

        if result["sanity_review_candidates"]:
            self._sanity_review_pending[rate] = result
            self._sanity_review_context[rate] = "regen"
            return {"type": "sanity_review",
                    "payload": {"rates": [{"rate": rate, "candidates": self._sanity_candidates_payload(result)}]}}

        if result["text"] is None:
            print(f"\nRegeneration at {rate}% failed -- no condensation accepted. "
                  "Previous outputs for this rate, if any, are unchanged.")
            self._regen_rate = None
            return None
        if result.get("sanity_failed"):
            print(f"\n{rate}%: no trial passed the sanity checks -- using the closest trial by word count anyway, flag it for a manual look.")
        elif not result["success"]:
            note = "using the closest trial instead of discarding it" if result["soft_accept"] else "no valid trial"
            print(f"\n{rate}%: no trial hit the target word count within tolerance -- {note}.")

        return self._step_regenerate_after_text(rate, result["text"])

    def _step_regenerate_after_text(self, rate, condensed_text):
        """Injection analysis for a regenerated condensation, shared by
        the immediately-accepted path and the sanity-review-accepted
        path (_step_apply_sanity_review) -- both end up here once a rate
        has a final, accepted text."""
        target_words = condense.target_word_count(self.text, rate)
        condensed_text, was_fixed, _, issues = condense.run_adversarial_review(
            condensed_text, self.ordered_sentences, target_words,
            self.corpus_name, self.adversarial_model, self.decisions2, rate,
        )
        if not hasattr(self, "adversarial_notices"):
            self.adversarial_notices = {}
        if not hasattr(self, "_adversarial_issues_by_rate"):
            self._adversarial_issues_by_rate = {}
        if was_fixed:
            self._adversarial_issues_by_rate[rate] = issues
            self.adversarial_notices[rate] = (
                f"Adversarial reviewer auto-fixed this condensation ({'; '.join(issues)}). "
                "The original text is preserved in the Phase 2 decision log."
            )
            print(f"{rate}%: {self.adversarial_notices[rate]}")
        else:
            self.adversarial_notices.pop(rate, None)
            self._adversarial_issues_by_rate.pop(rate, None)

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
            return {"type": "injection_review", "payload": {"rates": [{"rate": rate, "flags": self._flags_with_source_context(rate)}]}}

        return self._step_finish_regeneration()

    def _step_finish_regeneration(self):
        rate = self._regen_rate
        self._process_and_save_rate(rate, self.condensed_texts[rate])
        self._regen_rate = None
        print("\nPipeline complete.")
        return None
