"""Phase 3 pipeline: single-corpus 'Distant Reading' analyses plus a
Source-vs-Summary comparison against every approved condensation rate --
a Python-native, self-contained replacement for a researcher-supplied
Voyant/Spyral-based Phase 3 specification this repo doesn't use (no
external upload, no manual ID round-trips, no notebook-cell injection).

No interactive prompts live here -- prepare_phase3_context/
build_phase3_report are pure orchestration, taking an already-resolved
collocation-pair selection as a parameter. The interactive collocation
review itself (collocations.run_collocation_review, input()-based) is
called directly by run_pipeline.py, and its non-interactive equivalent
(collocations.select_pairs_by_index) by webapp/pipeline_session.py --
same "interactive wrapper vs. pure function" split as
phase1/pipeline.py's run_cluster_review vs.
PipelineSession.apply_cluster_review.
"""

from phase1.pipeline import TABLEAU20
from phase3 import collocations, comparison, distant_reading as dr, report, stopwords as sw


def assign_cluster_colors(clusterdefs, tableau20=TABLEAU20):
    return {c["name"]: tableau20[i % len(tableau20)] for i, c in enumerate(clusterdefs)}


def describe_collocation_skip_reason(clusterdefs, colloc_candidates, condensed_texts):
    """Returns a specific, human-readable reason the collocation-pair
    comparison (and its researcher decision) is being skipped, or None
    if it isn't being skipped -- either because a decision is actually
    needed, or because there's no condensation to compare against in the
    first place (not really a "skip," just not applicable). Shared by
    run_pipeline.py's console output, webapp/pipeline_session.py's log +
    manifest note, and the report's own Provenance section, so the same
    corpus never gets three different explanations for the same gap."""
    if not condensed_texts:
        return None
    if not clusterdefs:
        return (
            "no semantic clusters exist for this corpus (all were removed during Phase 1 "
            "cluster review), so there were no cluster-significant terms to search for a "
            "real collocation among"
        )
    if not colloc_candidates:
        return (
            "no real collocations were found among this corpus's cluster-significant terms "
            "-- none of the candidate terms actually co-occur within the proximity window"
        )
    return None


def prepare_phase3_context(phase1_state, source_text, nlp, stemmer):
    """Everything Phase 3 needs before any researcher decision: parses
    the source, builds the comprehensive stopword set, and scans for
    real source collocations. Returns a dict threaded through to
    build_phase3_report and the collocation-review step."""
    clusterdefs = phase1_state.get("clusterDefs", [])
    source_doc = nlp(source_text)
    stopword_set, auto_authors, author_candidates = sw.build_stopwords(nlp.Defaults.stop_words, source_doc)
    colloc_candidates = collocations.find_source_collocations(source_doc, clusterdefs, stemmer)
    return {
        "clusterdefs": clusterdefs,
        "colors_by_cluster": assign_cluster_colors(clusterdefs),
        "source_doc": source_doc,
        "stopwords": stopword_set,
        "auto_authors": auto_authors,
        "author_candidates": author_candidates,
        "colloc_candidates": colloc_candidates,
    }


def build_phase3_report(context, corpus_name, phase1_state, source_text, stemmer,
                         condensed_texts, selected_pairs, nlp, n_bins=5):
    """condensed_texts: {rate: condensed_text} for every approved rate
    (may be empty -- the single-corpus Distant Reading section still
    runs; only the Source-vs-Summary section is skipped). selected_pairs:
    the researcher's already-resolved collocation-pair choice (possibly
    empty). Returns the report's HTML string."""
    clusterdefs = context["clusterdefs"]
    colors_by_cluster = context["colors_by_cluster"]
    source_doc = context["source_doc"]
    stopword_set = context["stopwords"]

    reader_html = dr.build_reader_html(source_doc, clusterdefs, colors_by_cluster, stemmer)
    wordcloud_html = dr.build_wordcloud_html(source_text, stopword_set)
    bin_freqs = dr.bin_cluster_frequencies(source_text, clusterdefs, stemmer, n_bins)
    trend_chart_html = dr.build_trend_chart_svg(bin_freqs, colors_by_cluster)
    phrase_table_html = dr.build_phrase_table(source_doc, stopword_set)
    term_stats_html = dr.build_term_stats_table(source_doc, clusterdefs, stemmer)

    comparison_html = ""
    if condensed_texts:
        rate_docs = [(rate, text, nlp(text)) for rate, text in sorted(condensed_texts.items())]
        comparison_html = comparison.build_comparison_section(
            source_doc, rate_docs, phase1_state, source_text, stemmer, stopword_set, selected_pairs,
        )

    skip_reason = describe_collocation_skip_reason(clusterdefs, context["colloc_candidates"], condensed_texts)
    provenance_html = report.build_provenance_html(
        len(stopword_set), context["auto_authors"], context["colloc_candidates"], selected_pairs, skip_reason,
    )

    return report.build_distant_reading_report(
        corpus_name, clusterdefs, colors_by_cluster,
        reader_html, wordcloud_html, trend_chart_html, phrase_table_html, term_stats_html,
        comparison_html, provenance_html,
    )
