#!/usr/bin/env python
"""Runs Phase 0 (optional) -> Phase 1 -> Phase 2 -> Phase 3 (distant
reading + source-vs-summary comparison) -> standalone report export on
one new corpus in a single script.

Same interactive prompts, same outputs under data/<CORPUS_NAME>/, same
decision logs as running phase0.ipynb (optional) + phase1.ipynb +
phase2.ipynb by hand -- this script calls the exact same
phase1/pipeline.py, phase2/pipeline.py, phase2/condense.py,
phase2/condensation_report.py, phase3/pipeline.py,
common/standalone_report.py functions the notebooks call. Nothing here
is duplicated logic. See RUN_PIPELINE_GUIDE.md for a detailed usage guide.

Usage:
    python run_pipeline.py --corpus Boisseau --input raw.txt
    python run_pipeline.py --corpus Boisseau --input raw.txt --clean
    python run_pipeline.py --corpus Boisseau --input raw.txt --rates 10,20 --ollama-model llama3 --max-trials 3
"""

import argparse
import shutil
import tempfile
from pathlib import Path

import spacy
import torch
from nltk.stem import PorterStemmer

from common.paths import CorpusPaths
from common.decisions import DecisionLog
from common import standalone_report
from phase0.clean_corpus import clean_text
from phase1 import pipeline as phase1_pipeline
from phase2 import pipeline as phase2_pipeline, condense, condensation_report
from phase3 import pipeline as phase3_pipeline, collocations as phase3_collocations, report as phase3_report


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--corpus", required=True, help="Corpus name -- used for data/<corpus>/ paths")
    parser.add_argument("--input", required=True, help="Path to the raw .txt file for this corpus")
    parser.add_argument("--clean", action="store_true",
                         help="Run Phase 0 cleaning (phase0/clean_corpus.py) before Phase 1")

    # Phase 1 config -- same defaults as phase1.ipynb's config cell
    parser.add_argument("--top-percentile", type=float, default=0.50)
    parser.add_argument("--max-stems", type=int, default=150)
    parser.add_argument("--max-sentences-per-stem", type=int, default=5)
    parser.add_argument("--max-synsets", type=int, default=5,
                         help="Ceiling, not a guarantee -- candidates shown per flagged word are also "
                              "capped by how many senses WordNet actually has for that word's part of speech.")
    parser.add_argument("--max-cluster-size", type=int, default=10)
    parser.add_argument("--min-clusters", type=int, default=5)
    parser.add_argument("--min-cluster-len", type=int, default=3)
    parser.add_argument("--glossbert-model", default="jvomiranda/GlossBERT_Checkpoint")
    parser.add_argument("--lang-model", default="en_core_web_sm")
    parser.add_argument("--sentence-embedder", default="all-MiniLM-L6-v2")

    # Phase 2 config -- same defaults as phase2.ipynb's config cell
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--density-percentile", type=float, default=75)

    # Condensation config -- optional. Omit any of these to fall back to
    # the same interactive prompts phase2.ipynb uses
    # (condense.run_condensation_setup).
    parser.add_argument("--rates", help="Comma-separated condensation rate(s), e.g. '10,20'.")
    parser.add_argument("--ollama-model", help="Ollama model name.")
    parser.add_argument("--max-trials", type=int, help="Max generation trials per rate before escalation.")

    return parser.parse_args()


def place_raw_text(args, paths):
    if args.clean:
        with open(args.input, encoding="utf-8") as f:
            raw = f.read()
        cleaned = clean_text(raw)
        with open(paths.raw_txt(), "w", encoding="utf-8") as f:
            f.write(cleaned)
        print(f"Phase 0: cleaned corpus written to {paths.raw_txt()}")
    else:
        shutil.copy(args.input, paths.raw_txt())
        print(f"Copied raw corpus to {paths.raw_txt()}")


def resolve_condensation_config(args, decisions):
    """Uses --rates/--ollama-model/--max-trials if all three were given on
    the command line (still logged, marked "source": "cli" so the
    decision log stays a complete record regardless of run mode);
    otherwise falls back to the same interactive prompts phase2.ipynb
    uses."""
    if args.rates and args.ollama_model and args.max_trials:
        rates = [int(r.strip()) for r in args.rates.split(",") if r.strip()]

        decisions.record(
            step="condensation_setup", decision_type="rate_selection",
            prompt="What condensation rate(s) would you like? (5-30%)",
            choice=args.rates, extra={"parsed_rates": rates, "source": "cli"},
        )
        decisions.record(
            step="condensation_setup", decision_type="ollama_model_choice",
            prompt="Which Ollama model would you like to use?",
            choice=args.ollama_model, extra={"source": "cli"},
        )
        decisions.record(
            step="condensation_setup", decision_type="max_trials_choice",
            prompt="How many generation trials before asking to escalate?",
            choice=str(args.max_trials), extra={"source": "cli"},
        )

        print(
            f"\nWill generate condensations at {rates}% using '{args.ollama_model}', "
            f"up to {args.max_trials} trial(s) each before escalation."
        )
        return rates, args.ollama_model, args.max_trials

    return condense.run_condensation_setup(decisions)


def run_phase1(args, paths, decisions):
    """Mirrors phase1.ipynb cell-by-cell. Returns (phase1_state, nlp, stemmer, text)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    nlp = spacy.load(args.lang_model)
    stemmer = PorterStemmer()

    with open(paths.raw_txt(), "r", encoding="utf-8") as f:
        text = f.read()

    doc = nlp(text)

    top_stems = phase1_pipeline.extract_top_stems(doc, stemmer, args.top_percentile, args.max_stems)
    print(f"\nSelected {len(top_stems)} top stems\n")
    for stem, count in top_stems.items():
        print(f"{stem}: {count}")

    with open(paths.top_stems(), "w", encoding="utf-8") as out:
        out.write("stem\tcount\n")
        for stem, count in top_stems.items():
            out.write(f"{stem}\t{count}\n")
    print(f"\nSaved results to {paths.top_stems()}")

    print("\nLoading GlossBERT checkpoint...")
    tokenizer, model = phase1_pipeline.load_glossbert(args.glossbert_model, device)
    print("GlossBERT loaded successfully.")

    print("\nMapping stem occurrences...")
    stem_occurrences = phase1_pipeline.map_stems_to_sentences(doc, top_stems, stemmer)

    print("\nRunning GlossBERT analysis...")
    accepted_definitions, flagged_words = phase1_pipeline.run_glossbert_analysis(
        top_stems, stem_occurrences, tokenizer, model, device,
        max_sentences_per_stem=args.max_sentences_per_stem, max_synsets=args.max_synsets,
    )
    phase1_pipeline.print_accepted_definitions(accepted_definitions)
    phase1_pipeline.print_flagged_words(flagged_words)
    print("\nDone.")

    phase1_pipeline.run_flagged_term_review(flagged_words, accepted_definitions, decisions)
    print("\nSaving updated results...")
    phase1_pipeline.save_glossbert_output(accepted_definitions, paths.glossbert_output())
    print(f"\nUpdated results saved to {paths.glossbert_output()}")

    embedder, stem_texts, stem_names, embeddings = phase1_pipeline.build_stem_embeddings(
        accepted_definitions, args.sentence_embedder
    )
    labels, clusters = phase1_pipeline.cluster_stem_embeddings(
        embeddings, stem_names, args.min_clusters, min_cluster_len=args.min_cluster_len,
    )
    print(f"Stems embedded: {len(stem_names)}")
    print(f"Clusters found (excluding noise): {len(clusters)}")

    stem_word_frequencies = phase1_pipeline.build_stem_word_frequency_table(stem_occurrences)
    renamed_clusters = phase1_pipeline.rename_clusters(clusters, stem_word_frequencies)
    phase1_pipeline.print_named_clusters(renamed_clusters, "SEMANTIC CLUSTERS (BEFORE RECLUSTERING)")

    clusters, large_subclusters = phase1_pipeline.recluster_large_clusters(
        clusters, stem_occurrences, embedder, tokenizer, model, device,
        max_cluster_size=args.max_cluster_size, min_clusters=args.min_clusters,
        min_cluster_len=args.min_cluster_len, max_synsets=args.max_synsets,
    )
    phase1_pipeline.print_raw_clusters(clusters, "UPDATED CLUSTERS")
    phase1_pipeline.print_raw_clusters(large_subclusters, "LARGE CLUSTER SUBCLUSTERS")
    still_oversized = phase1_pipeline.find_oversized_clusters(clusters, args.max_cluster_size)
    if still_oversized:
        print(f"\nWARNING: {len(still_oversized)} cluster(s) are still over max_cluster_size "
              f"({args.max_cluster_size}) after splitting -- sizes: "
              f"{[len(set(v)) for v in still_oversized.values()]}. The data didn't separate further; "
              "review these in the cluster-review step below.")

    noise_stems = phase1_pipeline.extract_noise_stems(stem_names, labels)
    noise_clusters = phase1_pipeline.recluster_noise(
        noise_stems, stem_occurrences, embedder, tokenizer, model, device,
        min_clusters=args.min_clusters, min_cluster_len=args.min_cluster_len, max_synsets=args.max_synsets,
    )
    # recluster_noise's own HDBSCAN pass can just as easily produce an
    # oversized cluster as the primary pass does (its default
    # cluster-selection method favors one large stable cluster over many
    # small ones) -- run it through the same max_cluster_size split the
    # primary pass's output already gets, rather than letting it skip
    # that safety net purely because of which pass produced it.
    noise_clusters, noise_large_subclusters = phase1_pipeline.recluster_large_clusters(
        noise_clusters, stem_occurrences, embedder, tokenizer, model, device,
        max_cluster_size=args.max_cluster_size, min_clusters=args.min_clusters,
        min_cluster_len=args.min_cluster_len, max_synsets=args.max_synsets,
    )
    phase1_pipeline.print_raw_clusters(noise_clusters, "NOISE CLUSTERS")
    if noise_large_subclusters:
        phase1_pipeline.print_raw_clusters(noise_large_subclusters, "OVERSIZED NOISE-CLUSTER SUBCLUSTERS")
    still_oversized = phase1_pipeline.find_oversized_clusters(noise_clusters, args.max_cluster_size)
    if still_oversized:
        print(f"\nWARNING: {len(still_oversized)} noise-recovered cluster(s) are still over "
              f"max_cluster_size ({args.max_cluster_size}) after splitting -- sizes: "
              f"{[len(set(v)) for v in still_oversized.values()]}. The data didn't separate further; "
              "review these in the cluster-review step below.")

    stem_word_frequencies = phase1_pipeline.build_stem_word_frequency_table(stem_occurrences)
    renamed_clusters = phase1_pipeline.rename_clusters(clusters, stem_word_frequencies)
    renamed_noise_clusters = phase1_pipeline.rename_clusters(noise_clusters, stem_word_frequencies)
    renamed_clusters = phase1_pipeline.merge_named_clusters(renamed_clusters, renamed_noise_clusters)
    phase1_pipeline.print_named_clusters(renamed_clusters, "SEMANTIC CLUSTERS")

    sentence_cache = phase1_pipeline.build_sentence_token_cache(nlp, accepted_definitions, stemmer, use_lemmas=True)
    global_ngram_counter, percentile_threshold = phase1_pipeline.build_global_ngram_statistics(sentence_cache)
    all_clusters = {**clusters, **noise_clusters}
    cluster_ngrams = phase1_pipeline.extract_cluster_ngrams(
        all_clusters, accepted_definitions, sentence_cache,
        global_ngram_counter, percentile_threshold, stemmer, use_lemmas=True,
    )
    renamed_clusters = phase1_pipeline.attach_ngrams(renamed_clusters, cluster_ngrams)

    final_clusters, excluded_cluster_ngrams = phase1_pipeline.run_cluster_review(renamed_clusters, decisions)

    phase1_state = phase1_pipeline.build_phase1_state(final_clusters, excluded_cluster_ngrams)
    phase1_pipeline.save_phase1_state(phase1_state, paths.phase1_state_json())
    print(f"\nSaved JSON to {paths.phase1_state_json()}")

    html = phase1_pipeline.build_cluster_html(final_clusters, args.corpus)
    phase1_pipeline.save_html(html, paths.phase1_html())
    print(f"\nHTML cluster snippet written:\n{paths.phase1_html()}")

    return phase1_state, nlp, stemmer, text


def process_and_save_rate(rate, condensed_text, args, paths, decisions, phase1_state, nlp, stemmer,
                           text, enriched_state, title=None, authors=None, date=None):
    """Runs injection analysis -> borderline review -> cluster coverage ->
    report building -> saving (condensed text, injection report, HTML
    fragment/preview, human report, plain summary, standalone report),
    for one already-generated rate. Shared by the
    initial per-rate loop in run_phase2 and the post-condensation
    regeneration loop, so a regenerated rate goes through the exact same
    pipeline as a freshly generated one. Returns the HTML fragment."""
    condensed_path = paths.condensation_paths(rate)["condensed_text"]
    with open(condensed_path, "w", encoding="utf-8") as f:
        f.write(condensed_text)
    print(f"\nSaved condensed text to {condensed_path}")

    print(f"\n=== Injection analysis: {rate}% condensation ===")
    all_spans, source_sentences = condense.classify_condensation(condensed_text, text, nlp)
    stats = condense.compute_injection_stats(all_spans, condensed_text, text)
    borderline_flags = condense.flag_borderline_classifications(all_spans)

    print(f"Spans classified: {len(all_spans)}")
    print(f"Non-injected word share: {stats['non_injected_pct']}%")
    print(f"Verbatim-overlap scan: {stats['verbatim_overlap_pct']}%")

    condense.run_injection_review(all_spans, borderline_flags, rate, decisions, source_sentences=source_sentences)

    coverage = condense.compute_cluster_coverage(phase1_state, condensed_text, text, stemmer)
    print(f"\n=== Cluster coverage: {rate}% condensation ===")
    for name, r in coverage.items():
        print(f"  {name}: target {r['target']}%, actual {r['actual']}% ({r['delta']:+.1f}pp) -- {r['status']}")

    fragment = condensation_report.build_condensation_fragment(
        condensed_text, all_spans, source_sentences, coverage,
        corpus_name=args.corpus, rate=rate,
        source_word_count=condense.count_words(text),
        phase1_json_name=paths.phase1_state_json().name,
        title=title, authors=authors, date=date,
    )
    preview = condensation_report.build_standalone_preview(fragment, args.corpus)
    ordered_sentences = condense.gather_ordered_informative_sentences(enriched_state)
    sanity_issues = condense.check_condensation_sanity(condensed_text, ordered_sentences)
    report = condensation_report.build_human_report(
        all_spans, borderline_flags, coverage,
        stats["verbatim_overlap_pct"], stats["non_injected_pct"], sanity_issues,
    )
    blocks = condensation_report.parse_condensed_blocks(condensed_text)
    summary = condensation_report.build_plain_summary(blocks)

    injection_report_data = {
        "corpus": args.corpus, "rate": rate,
        "spans": all_spans, "borderline_flags": borderline_flags,
        "coverage": coverage, "stats": stats,
    }

    output_paths = paths.condensation_paths(rate)
    condensation_report.save_condensation_outputs(
        output_paths, fragment, preview, report, summary, injection_report_data,
    )

    print(f"\n=== {rate}% condensation outputs ===")
    for label, p in output_paths.items():
        if label != "condensed_text":
            print(f"  {label}: {p}")

    report_html = standalone_report.build_standalone_report(
        enriched_state, fragment, args.corpus, rate, paths,
    )
    report_path = paths.standalone_report_path(rate)
    standalone_report.save_standalone_report(report_html, report_path)
    print(f"Standalone report written to {report_path}")

    return fragment


def edit_prompt_via_tempfile(default_prompt):
    """Writes the rendered default condensation prompt to a scratch file,
    tells the researcher to open/edit/save it in their own editor, waits
    for ENTER, then reads it back -- avoids needing multi-line terminal
    input for what can be a long, multi-paragraph prompt."""
    edit_path = Path(tempfile.gettempdir()) / "peel_condensation_prompt.txt"
    edit_path.write_text(default_prompt, encoding="utf-8")
    print(
        f"\nDefault prompt written to:\n  {edit_path}\n"
        "Open it in a text editor, make any changes, save the file, then press "
        "ENTER here to continue (leave it unchanged to use the default as-is)."
    )
    input()
    return edit_path.read_text(encoding="utf-8")


def run_regeneration_loop(condensed_texts, args, paths, decisions, phase1_state, nlp, stemmer, text,
                           enriched_state, ordered_sentences, cluster_key_terms, ollama_model, max_trials,
                           title=None, authors=None, date=None):
    """After all requested rates have been generated and their reports
    built, offers to regenerate one -- optionally at a different target
    rate, optionally with a hand-edited prompt -- for as many rounds as
    the researcher wants, before the script ends (and they'd next run it
    again for a new corpus)."""
    while True:
        regenerate = input("\nWould you like to regenerate a condensation? (y/n): ").strip().lower()
        decisions.record(
            step="condensation_regeneration", decision_type="regenerate_prompt",
            prompt="Would you like to regenerate a condensation?",
            options=["y", "n"], choice=regenerate,
        )
        if regenerate != "y":
            break

        existing_rates = sorted(condensed_texts.keys())
        rate_input = input(
            f"\nExisting rates: {', '.join(f'{r}%' for r in existing_rates) or 'none'}\n"
            "Which rate would you like to regenerate? (enter one of the above to redo it, "
            "or a new percentage to add another): "
        ).strip()
        try:
            rate = int(rate_input)
        except ValueError:
            print("Invalid rate -- skipping.")
            continue

        decisions.record(
            step="condensation_regeneration", decision_type="rate_selection",
            prompt="Which rate would you like to regenerate?",
            options=[str(r) for r in existing_rates], choice=rate_input,
            extra={"parsed_rate": rate},
        )

        adjust = input(f"\nAdjust the condensation rate? Currently targeting {rate}%. (y/n): ").strip().lower()
        decisions.record(
            step="condensation_regeneration", decision_type="adjust_rate_prompt",
            prompt="Adjust the condensation rate?", options=["y", "n"], choice=adjust,
            extra={"rate": rate},
        )
        if adjust == "y":
            new_rate_input = input("New target rate (%): ").strip()
            try:
                rate = int(new_rate_input)
            except ValueError:
                print("Invalid rate -- keeping the original.")
            decisions.record(
                step="condensation_regeneration", decision_type="new_rate_value",
                prompt="New target rate (%)", choice=new_rate_input,
                extra={"resolved_rate": rate},
            )

        target_words = condense.target_word_count(text, rate)
        default_prompt = condense.build_condensation_prompt(
            ordered_sentences, cluster_key_terms, target_words, args.corpus,
        )

        add_info = input(
            "\nAdd information to the LLM by editing the default prompt? (y/n): "
        ).strip().lower()
        decisions.record(
            step="condensation_regeneration", decision_type="edit_prompt_prompt",
            prompt="Add information to the LLM by editing the default prompt?",
            options=["y", "n"], choice=add_info, extra={"rate": rate},
        )
        prompt_text = default_prompt
        if add_info == "y":
            prompt_text = edit_prompt_via_tempfile(default_prompt)
            decisions.record(
                step="condensation_regeneration", decision_type="prompt_edited",
                prompt="Edited condensation prompt", choice="(see extra.final_prompt)",
                extra={
                    "rate": rate, "prompt_changed": prompt_text != default_prompt,
                    "final_prompt": prompt_text,
                },
            )

        print(f"\n=== Regenerating {rate}% condensation (target ~{target_words} words) ===")
        result = condense.attempt_condensation_trials(
            ordered_sentences, cluster_key_terms, target_words, args.corpus,
            model=ollama_model, max_trials=max_trials, include_full_text=False,
            prompt_override=prompt_text,
        )
        for t in result["trials"]:
            print(condense.format_trial_line(t))
            decisions.record(
                step="condensation_regeneration", decision_type="trial_result",
                prompt="Condensation regeneration trial", extra={"rate": rate, **t},
            )

        if result["sanity_review_candidates"]:
            result = condense.run_sanity_review_for_rate(rate, result, decisions)

        if result["text"] is None:
            print(f"\nRegeneration at {rate}% failed -- no condensation accepted. Previous outputs for this rate, if any, are unchanged.")
            continue
        if result.get("researcher_accepted_despite_sanity"):
            pass  # already reported by run_sanity_review_for_rate above
        elif result.get("sanity_failed"):
            print(f"\n{rate}%: no trial passed the sanity checks -- using the closest trial by word count anyway, flag it for a manual look.")
        elif not result["success"]:
            note = "using the closest trial instead of discarding it" if result["soft_accept"] else "no valid trial"
            print(f"\n{rate}%: no trial hit the target word count within tolerance -- {note}.")

        condensed_texts[rate] = result["text"]
        process_and_save_rate(
            rate, result["text"], args, paths, decisions, phase1_state, nlp, stemmer, text, enriched_state,
            title=title, authors=authors, date=date,
        )
        print(f"\nRegenerated {rate}% condensation.")


def run_phase3(args, paths, phase1_state, nlp, stemmer, text, condensed_texts):
    """Runs Phase 3 (single-corpus distant reading, plus a source-vs-summary
    comparison for every rate already generated by the time this runs --
    a later post-completion regeneration does not retroactively update
    this report). Own decision log, own output file. Mirrors
    phase3/pipeline.py's pure orchestration; the collocation-pair
    selection is the one interactive step, handled here via
    collocations.run_collocation_review rather than inside phase3/pipeline.py
    itself -- same "interactive wrapper vs. pure function" split as
    run_phase1's use of phase1_pipeline.run_cluster_review."""
    decisions3 = DecisionLog(args.corpus, phase="phase3")

    print("\n===================================")
    print("PHASE 3: DISTANT READING")
    print("===================================\n")

    context = phase3_pipeline.prepare_phase3_context(phase1_state, text, nlp, stemmer)
    print(
        f"{len(context['stopwords'])} stopword(s) in effect "
        f"({len(context['auto_authors'])} author name(s) auto-detected)."
    )

    selected_pairs = []
    if condensed_texts and context["colloc_candidates"]:
        selected_pairs = phase3_collocations.run_collocation_review(context["colloc_candidates"], decisions3)
    else:
        skip_reason = phase3_pipeline.describe_collocation_skip_reason(
            context["clusterdefs"], context["colloc_candidates"], condensed_texts,
        )
        if skip_reason:
            print(f"\nContexts/Collocates comparison skipped: {skip_reason}.")

    html = phase3_pipeline.build_phase3_report(
        context, args.corpus, phase1_state, text, stemmer, condensed_texts, selected_pairs, nlp,
    )
    phase3_report.save_distant_reading_report(html, paths.distant_reading_report_path())
    print(f"\nDistant reading report written to {paths.distant_reading_report_path()}")


def run_phase2(args, paths, decisions, phase1_state, nlp, stemmer, text):
    """Mirrors phase2.ipynb cell-by-cell."""
    enriched_state = phase2_pipeline.enrich_with_informative_sentences(
        phase1_state, text, nlp, stemmer,
        top_n=args.top_n, density_percentile=args.density_percentile,
    )
    phase2_pipeline.save_informative_sentences(enriched_state, paths.phase2_output_json())
    print(f"\nSaved enriched JSON to\n{paths.phase2_output_json()}")

    rates, ollama_model, max_trials = resolve_condensation_config(args, decisions)

    # Metadata detection reads the untouched original input, not `text`
    # (which may already be Phase-0-cleaned) -- title/author/date front
    # matter is reliably positioned at the very start of the document,
    # and Phase 0 cleaning isn't guaranteed to leave it alone (e.g. a
    # title that also repeats as a running header gets stripped as a
    # recurring short line).
    with open(args.input, encoding="utf-8") as f:
        original_text = f.read()
    title, authors, date = condense.run_source_metadata_setup(decisions, original_text, ollama_model)

    ordered_sentences = condense.gather_ordered_informative_sentences(enriched_state)
    cluster_key_terms = condense.gather_cluster_key_terms(enriched_state)

    condensed_texts = {}
    for rate in rates:
        target_words = condense.target_word_count(text, rate)
        result = condense.run_generate_for_rate(
            rate, ordered_sentences, cluster_key_terms, target_words, args.corpus,
            text, ollama_model, max_trials, decisions,
        )
        if result["text"] is None:
            continue
        condensed_texts[rate] = result["text"]

    for rate, condensed_text in condensed_texts.items():
        process_and_save_rate(
            rate, condensed_text, args, paths, decisions, phase1_state, nlp, stemmer, text, enriched_state,
            title=title, authors=authors, date=date,
        )

    run_phase3(args, paths, phase1_state, nlp, stemmer, text, condensed_texts)

    run_regeneration_loop(
        condensed_texts, args, paths, decisions, phase1_state, nlp, stemmer, text,
        enriched_state, ordered_sentences, cluster_key_terms, ollama_model, max_trials,
        title=title, authors=authors, date=date,
    )


def main():
    args = parse_args()

    paths = CorpusPaths(args.corpus)
    paths.ensure_dirs()

    place_raw_text(args, paths)

    decisions1 = DecisionLog(args.corpus, phase="phase1")
    phase1_state, nlp, stemmer, text = run_phase1(args, paths, decisions1)

    decisions2 = DecisionLog(args.corpus, phase="phase2")
    run_phase2(args, paths, decisions2, phase1_state, nlp, stemmer, text)

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
