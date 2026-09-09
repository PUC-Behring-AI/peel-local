"""Empirical collocation-pair discovery for the source-vs-summary
Contexts comparison: scans the real source text for pairs of
cluster-significant terms (drawn from different clusters) that actually
co-occur within a proximity window, ranks candidates by real hit count,
and flags any pair where one term's own frequency is disproportionate --
a possible base-rate confound, not a real relationship.

Adapted from a researcher-supplied specification for a separate,
Voyant-based PEEL system this repo doesn't use, which found (after two
earlier, deterministic designs) that a term pair must actually be
observed to co-occur before it's worth comparing across source and
summary -- picking two "important-looking" terms without checking
usually produced a pair that never appears near each other at all.

The researcher always picks from these ranked, flagged candidates
(run_collocation_review, or the webapp's equivalent
select_pairs_by_index) -- never an automatic top-score pick, since the
confound check is a heuristic, not a verdict.
"""

from phase3 import terms as pterms


def _cluster_term_pool(clusterdefs, stems, lemmas, top_terms_per_cluster=6):
    """For every cluster, its stems/n-grams actually present in the
    source at least once, ranked by source frequency and capped -- the
    candidate pool collocation pairs are drawn from."""
    pool = []
    for cluster in clusterdefs:
        scored = []
        for term in list(cluster.get("stems", [])) + list(cluster.get("ngrams", [])):
            count = len(pterms.term_positions(stems, lemmas, term))
            if count > 0:
                scored.append((term, count))
        scored.sort(key=lambda x: x[1], reverse=True)
        for term, count in scored[:top_terms_per_cluster]:
            pool.append({"term": term, "cluster": cluster["name"], "count": count})
    return pool


def find_source_collocations(doc, clusterdefs, stemmer, proximity_n=5,
                              top_terms_per_cluster=6, top_n_candidates=20):
    """doc: a spaCy Doc of the source text. Returns a list of candidate
    dicts (term_a, cluster_a, term_b, cluster_b, hits, confounded,
    confound_reason), ranked by hit count, most co-occurring first."""
    stems, lemmas = pterms.token_forms(doc, stemmer)
    pool = _cluster_term_pool(clusterdefs, stems, lemmas, top_terms_per_cluster)
    if len(pool) < 2:
        return []

    positions_by_term = {item["term"]: pterms.term_positions(stems, lemmas, item["term"]) for item in pool}

    candidates = []
    seen_pairs = set()
    for i, a in enumerate(pool):
        for b in pool[i + 1:]:
            if a["cluster"] == b["cluster"]:
                continue
            pair_key = tuple(sorted((a["term"], b["term"])))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            hits = pterms.count_positions_near(
                positions_by_term[a["term"]], positions_by_term[b["term"]], proximity_n,
            )
            if hits == 0:
                continue
            candidates.append({
                "term_a": a["term"], "cluster_a": a["cluster"],
                "term_b": b["term"], "cluster_b": b["cluster"],
                "hits": hits,
            })

    counts = sorted(item["count"] for item in pool)
    median_count = counts[len(counts) // 2] if counts else 0
    count_by_term = {item["term"]: item["count"] for item in pool}
    for c in candidates:
        dominant = max(count_by_term[c["term_a"]], count_by_term[c["term_b"]])
        c["confounded"] = median_count > 0 and dominant >= 5 * median_count
        c["confound_reason"] = (
            f"one term occurs {dominant}x vs. a corpus median of {median_count}x among "
            "candidate terms -- co-occurrence may just be base rate" if c["confounded"] else ""
        )

    candidates.sort(key=lambda c: c["hits"], reverse=True)
    return candidates[:top_n_candidates]


def format_collocation_candidates(candidates):
    lines = []
    for i, c in enumerate(candidates, 1):
        flag = "  [possibly confounded]" if c["confounded"] else ""
        lines.append(
            f"{i}. {c['term_a']} ({c['cluster_a']})  <->  {c['term_b']} ({c['cluster_b']})"
            f"  -- {c['hits']} co-occurrence(s){flag}"
        )
    return "\n".join(lines)


def select_pairs_by_index(candidates, indices):
    """indices: iterable of 0-based ints. Falls back to the single
    top-ranked candidate on an empty/out-of-range selection -- never
    returns nothing when candidates exist, so the comparison section
    always has at least one pair to show when one is available. Shared
    by both the CLI's run_collocation_review and the webapp's
    apply_collocation_review, so there's one resolution rule, not two."""
    selected = [candidates[i] for i in sorted(set(indices)) if 0 <= i < len(candidates)]
    return selected or (candidates[:1] if candidates else [])


def run_collocation_review(candidates, decisions):
    """Interactive: presents ranked, confound-flagged collocation
    candidates and lets the researcher pick one or more pairs to compare
    (via Contexts/Collocates) between the source and each condensation
    summary. Every choice is appended to the decisions log via
    `decisions.record(...)` for later audit -- it does not change the
    interactive flow. Shared by run_pipeline.py; the webapp calls
    select_pairs_by_index directly instead, same pattern as
    phase1/pipeline.py's run_cluster_review vs.
    PipelineSession.apply_cluster_review."""
    if not candidates:
        print("\nNo real source collocations found among the cluster-significant terms.")
        return []

    print("\n===================================")
    print("COLLOCATION-PAIR SELECTION")
    print("===================================\n")
    print("These term pairs actually co-occur in the source text. Pick one or")
    print("more to compare between the source and each summary in the report.\n")
    print("How candidates are found: each Phase 1 cluster's top 6 stems/n-grams by source")
    print("frequency form a term pool; every pair from two different clusters that actually")
    print("sits within 5 tokens of each other somewhere in the source becomes a candidate,")
    print("ranked by co-occurrence count and capped at the top 20 shown below.\n")
    print("What 'possibly confounded' means: unrelated to co-occurrence count. A pair is")
    print("flagged when its more frequent term occurs, on its own, at least 5x the median")
    print("occurrence count across the whole term pool -- that common a term will sit near")
    print("almost anything, so the count is weaker evidence, not necessarily wrong. It's a")
    print("caveat for your judgment, not a verdict -- pick a flagged pair if it still looks")
    print("meaningful.\n")
    print(format_collocation_candidates(candidates))

    raw = input(
        "\nType numbers to select (comma-separated), 'all' for every candidate, "
        "or ENTER for the top-ranked pair: "
    ).strip()

    if not raw:
        indices = {0}
    elif raw.lower() in ("all", "*"):
        indices = set(range(len(candidates)))
    else:
        try:
            indices = {int(x.strip()) - 1 for x in raw.split(",") if x.strip()}
        except ValueError:
            print("Invalid input -- using the top-ranked pair.")
            indices = {0}

    selected = select_pairs_by_index(candidates, indices)

    decisions.record(
        step="collocation_review", decision_type="pair_selection",
        prompt="Which collocation pair(s) to compare between source and summaries?",
        options=[f"{c['term_a']} <-> {c['term_b']}" for c in candidates], choice=raw,
        extra={"selected": [f"{c['term_a']} <-> {c['term_b']}" for c in selected]},
    )
    return selected
