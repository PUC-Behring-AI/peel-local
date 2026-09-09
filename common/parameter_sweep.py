"""Phase 1 parameter sweep: previews the effect of several candidate
`frequency_percentile` values before committing to one, without running any
of the expensive GPU-bound machinery (GlossBERT, embeddings, HDBSCAN).

Deliberately lives here rather than in phase1/pipeline.py or
phase2/pipeline.py: it calls into both (extract_top_stems/
map_stems_to_sentences from Phase 1, segment_corpus/compute_lexical_densities/
select_representative_sentences from Phase 2), and common/ is where
cross-phase orchestration already lives (see standalone_report.py, which
reads both phases' state).

The Phase 2 functions reused here have no dependency on real Phase 1
clusters existing -- select_representative_sentences only ever reads
cluster["stems"]/cluster["ngrams"] from whatever dict it's handed, so a
single synthetic pseudo-cluster wrapping a candidate's top_stems produces
a correct, unmodified answer.
"""

from phase1 import pipeline as phase1_pipeline
from phase2 import pipeline as phase2_pipeline

DEFAULT_FREQUENCY_PERCENTILE_CANDIDATES = (0.2, 0.35, 0.5, 0.65, 0.8)
DEFAULT_DENSITY_PERCENTILES = (50, 75, 90)


def compute_sweep_report(doc, text, nlp, stemmer,
                          candidates=DEFAULT_FREQUENCY_PERCENTILE_CANDIDATES,
                          density_percentiles=DEFAULT_DENSITY_PERCENTILES,
                          word_frequency_percentile=0.0):
    """Pure computation, no I/O -- safe to call from a background thread
    (webapp) as well as interactively (CLI/notebook, via
    run_parameter_sweep_setup below).

    Deliberately ignores max_stems: extract_top_stems's own cap
    (`final_n = min(ceil(vocab_size * (1 - frequency_percentile)), max_stems)`)
    would otherwise silently clip every candidate down to the same ceiling,
    hiding how many stems a given frequency_percentile *actually* selects --
    exactly the number a researcher needs to judge the parameter by. Each
    row reports the true, uncapped stem count for its frequency_percentile;
    max_stems is applied separately, after a percentile is chosen (see
    run_parameter_sweep_setup's follow-up prompt), the same way it always
    was outside the sweep.

    `word_frequency_percentile` applies filter_stem_occurrences_by_word_
    frequency to each candidate's occurrences BEFORE the sentence-coverage/
    informativeness counts are computed, using whatever value the
    researcher has already configured (CLI flag / notebook constant /
    webapp field) -- so the preview reflects the real corpus that would
    reach GlossBERT once that filter is applied downstream, not a
    pre-filter picture. 0 (the default) is a no-op, same as everywhere
    else this parameter appears.

    Returns one dict per candidate: {frequency_percentile, n_stems,
    n_distinct_words, n_sentences_with_stem_hit, n_sentences_total,
    informative_counts: {percentile: count}}."""
    _, sentence_docs = phase2_pipeline.segment_corpus(nlp, text)
    n_sentences_total = len(sentence_docs)

    density_bands = {}
    for p in density_percentiles:
        densities, threshold = phase2_pipeline.compute_lexical_densities(sentence_docs, p)
        density_bands[p] = (densities, threshold)

    rows = []
    for frequency_percentile in candidates:
        # max_stems=10**9 is "no cap" -- extract_top_stems always takes
        # min(vocab-fraction, max_stems), so an effectively-unbounded
        # ceiling here just reports the vocab-fraction count as-is.
        top_stems = phase1_pipeline.extract_top_stems(doc, stemmer, frequency_percentile, max_stems=10**9)
        stem_occurrences = phase1_pipeline.map_stems_to_sentences(doc, top_stems, stemmer)
        if word_frequency_percentile > 0:
            stem_occurrences = phase1_pipeline.filter_stem_occurrences_by_word_frequency(
                stem_occurrences, word_frequency_percentile,
            )
        n_distinct_words = len({occ["word"] for occs in stem_occurrences.values() for occ in occs})
        n_sentences_with_stem_hit = len({
            occ["sentence"] for occs in stem_occurrences.values() for occ in occs
        })

        pseudo_cluster = {"stems": list(top_stems.keys()), "ngrams": []}
        informative_counts = {}
        for p, (densities, threshold) in density_bands.items():
            selected = phase2_pipeline.select_representative_sentences(
                pseudo_cluster, sentence_docs, densities, threshold,
                stemmer, top_n=n_sentences_total,
            )
            informative_counts[p] = len(selected)

        rows.append({
            "frequency_percentile": frequency_percentile,
            "n_stems": len(top_stems),
            "n_distinct_words": n_distinct_words,
            "n_sentences_with_stem_hit": n_sentences_with_stem_hit,
            "n_sentences_total": n_sentences_total,
            "informative_counts": informative_counts,
        })

    return rows


def format_sweep_report(rows):
    """Plain-text comparison table, one row per candidate, for CLI/notebook
    printing (webapp renders `rows` itself as HTML cards)."""
    density_percentiles = sorted(rows[0]["informative_counts"].keys()) if rows else []
    header = (
        f"{'#':>2}  {'freq_percentile':>15}  {'n_stems':>8}  {'distinct words':>14}  "
        f"{'sentences w/ stem hit':>22}  " +
        "  ".join(f"{'informative @' + str(p) + 'th':>18}" for p in density_percentiles)
    )
    lines = [header, "-" * len(header)]
    for i, row in enumerate(rows, start=1):
        hit_col = f"{row['n_sentences_with_stem_hit']}/{row['n_sentences_total']}"
        info_cols = "  ".join(
            f"{row['informative_counts'][p]:>18}" for p in density_percentiles
        )
        lines.append(
            f"{i:>2}  {row['frequency_percentile']:>15}  {row['n_stems']:>8}  "
            f"{row['n_distinct_words']:>14}  {hit_col:>22}  {info_cols}"
        )
    return "\n".join(lines)


def run_parameter_sweep_setup(doc, text, nlp, stemmer, max_stems, decisions,
                               candidates=DEFAULT_FREQUENCY_PERCENTILE_CANDIDATES,
                               density_percentiles=DEFAULT_DENSITY_PERCENTILES,
                               word_frequency_percentile=0.0):
    """Interactive: computes the sweep, prints a comparison table, and lets
    the researcher pick a candidate by number or type a custom
    frequency_percentile (and optionally a custom max_stems). Every choice
    is appended to the decisions log via `decisions.record(...)` for later
    audit -- it does not change the interactive flow. Shared by
    phase1.ipynb and run_pipeline.py so there's one implementation, not
    two. Returns (frequency_percentile, max_stems).

    `word_frequency_percentile` is whatever the researcher already
    configured (CLI flag / notebook constant) for the separate
    derived-word filter -- passed straight through to compute_sweep_report
    so the previewed counts already reflect it (not swept as its own axis
    here; changing it means re-running the sweep, same as changing
    max_stems does)."""
    print("\n===================================")
    print("PARAMETER SWEEP: frequency_percentile candidates")
    print("===================================\n")
    print(
        "frequency_percentile is a statistical percentile CUTOFF on the "
        "corpus's frequency-ranked stem vocabulary -- a LARGER value is a "
        "stricter bar and keeps FEWER, more frequent stems; a SMALLER value "
        "keeps MORE stems, reaching further into the rarer, longer tail. "
        "E.g. 0.75 keeps only the top 25% most-frequent stems, matching the "
        "everyday sense of a percentile cutoff (like '90th percentile' "
        "meaning 'top 10%'). Counts below are uncapped by max_stems, so you "
        "can see each candidate's true size before deciding on a ceiling."
    )
    if word_frequency_percentile > 0:
        print(
            f"\nword_frequency_percentile={word_frequency_percentile} is already configured -- "
            "the 'distinct words' and sentence-coverage counts below already reflect that "
            "derived-word filter, not the unfiltered corpus."
        )
    print()

    rows = compute_sweep_report(
        doc, text, nlp, stemmer, candidates, density_percentiles, word_frequency_percentile,
    )
    print(format_sweep_report(rows))

    decisions.record(
        step="phase1_parameter_sweep", decision_type="sweep_shown",
        prompt="Frequency-percentile sweep candidates", choice=None,
        extra={"candidates": rows, "max_stems": max_stems, "word_frequency_percentile": word_frequency_percentile},
    )

    choice = input(
        f"\nPick a candidate number (1-{len(rows)}), or type a custom "
        "frequency_percentile (e.g. '0.55'): "
    ).strip()

    if choice.isdigit() and 1 <= int(choice) <= len(rows):
        chosen_row = rows[int(choice) - 1]
        frequency_percentile = chosen_row["frequency_percentile"]
        is_custom = False
    else:
        try:
            frequency_percentile = float(choice)
        except ValueError:
            print(f"Invalid choice {choice!r} -- keeping the default (candidate 1).")
            frequency_percentile = rows[0]["frequency_percentile"]
        is_custom = True

    uncapped_n_stems = next(
        (r["n_stems"] for r in rows if r["frequency_percentile"] == frequency_percentile), None
    )
    if uncapped_n_stems is not None and uncapped_n_stems > max_stems:
        print(f"\nNote: frequency_percentile={frequency_percentile} selects {uncapped_n_stems} stems "
              f"uncapped; max_stems={max_stems} will cap the actual run to {max_stems}.")

    max_stems_input = input(
        f"\nCustom max_stems? (ENTER to keep {max_stems}): "
    ).strip()
    if max_stems_input:
        try:
            max_stems = int(max_stems_input)
        except ValueError:
            print(f"Invalid value {max_stems_input!r} -- keeping max_stems={max_stems}.")

    decisions.record(
        step="phase1_parameter_sweep", decision_type="frequency_percentile_choice",
        prompt=f"Pick a candidate number (1-{len(rows)}), or type a custom frequency_percentile",
        options=[str(r["frequency_percentile"]) for r in rows], choice=choice,
        extra={"resolved_frequency_percentile": frequency_percentile, "resolved_max_stems": max_stems, "custom": is_custom},
    )

    print(f"\nUsing frequency_percentile={frequency_percentile}, max_stems={max_stems}.")
    return frequency_percentile, max_stems
