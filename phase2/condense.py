"""Phase 2 condensation: generates an LLM-condensed version of the corpus
at a target word-count rate, then verifies it -- a verbatim-overlap scan,
a deterministic F/T/R/C injection-risk classification of every non-verbatim
sentence, a borderline-classification flag pass, and per-cluster term
coverage against Phase 1's clusters.

Adapted from a researcher-supplied specification for a separate, more
elaborate PEEL system this repo doesn't implement. Two deliberate
adaptations from that spec:

1. Classification is **pure deterministic Python heuristics**, not an LLM
   judgment call -- reproducible, but cruder than a semantic reading.
   Ambiguous cases default to the more conservative label (C over R, T
   over F), same as the source spec's own tie-break rule.
2. The source spec's classification procedure mixes token-level verbatim
   scanning with sentence-level classification rules, which only coheres
   if an LLM is doing the actual span segmentation by reading the text --
   not viable under deterministic heuristics. Here, classification runs
   at the sentence level throughout: `scan_verbatim_overlap` still reports
   the token-level aggregate statistic, but `classify_condensation` makes
   one F/T/R/C decision per condensed sentence.

Interactive control flow (rate/model/trial-count prompts, the escalation
question, borderline-reclassification prompts) lives here as
`run_condensation_setup`/`run_generate_for_rate`/`run_injection_review`,
so phase2.ipynb and run_pipeline.py share one implementation instead of
two -- same pattern as phase1/pipeline.py's `run_*` functions.
"""

import re

from common import ollama_client
from phase2.condensation_report import parse_condensed_blocks

FUNCTION_WORD_SET = {
    "a", "an", "the", "this", "that", "these", "those",
    "and", "or", "but", "if", "then", "so", "nor",
    "of", "in", "on", "at", "by", "for", "with", "without", "to", "from",
    "as", "is", "are", "was", "were", "be", "been", "being",
    "it", "its", "we", "they", "he", "she", "which", "who", "what",
    "i", "you", "your", "my", "our", "one",
}

# Nouns that are technically content-bearing by part of speech but function
# as pure discourse connectives in phrases like "by contrast", "in sum",
# "in short", "in turn". classify_span's F rule is POS/dependency-based
# now (see CONNECTIVE_POS/_is_connective_token below) and no longer reads
# this list directly; it's still used by flag_borderline_classifications'
# lexical cross-check and by general content-word counting (e.g. cluster
# coverage), where "contrast" etc. should still count normally.
DISCOURSE_MARKER_EXTRAS = {"contrast", "sum", "short", "turn", "addition", "regard", "view"}

METALINGUISTIC_MARKERS = (
    "this paper", "this section", "this study", "this article", "this chapter",
    "the paper", "the article", "the following section", "the following sections",
    "these considerations", "this argument", "the argument", "as we have seen",
    "in what follows", "we now turn", "we conclude", "to summarize", "in sum",
    "this discussion", "the discussion above", "the preceding", "the remainder of",
)


# ============================================================
# GATHERING INPUTS FROM PHASE 1 / PHASE 2
# ============================================================

def gather_ordered_informative_sentences(phase1_state):
    """Flattens every cluster's most_insightful_sentences, deduped and
    sorted by corpus sentence index, so the LLM sees them in corpus order."""
    seen = {}
    for cluster in phase1_state.get("clusterDefs", []):
        for item in cluster.get("most_insightful_sentences", []):
            seen[item["index"]] = item
    return [seen[idx] for idx in sorted(seen)]


def gather_cluster_key_terms(phase1_state):
    return {
        cluster["name"]: list(cluster.get("stems", [])) + list(cluster.get("ngrams", []))
        for cluster in phase1_state.get("clusterDefs", [])
    }


# ============================================================
# CONDENSATION GENERATION
# ============================================================

def count_words(text):
    return len(text.split())


CTX_BUCKETS = (2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144)


def estimate_num_ctx(prompt, target_words, buckets=CTX_BUCKETS):
    """Rounds the prompt + expected-output size up to the next context
    bucket, instead of letting Ollama fall back to the model's own max
    context length (which can be far larger than needed and force the
    KV cache off the GPU -- see ollama_client.generate)."""
    estimated_tokens = (count_words(prompt) + target_words) * 1.4
    for bucket in buckets:
        if estimated_tokens <= bucket:
            return bucket
    return buckets[-1]


def target_word_count(source_text, rate_pct):
    return max(1, round(count_words(source_text) * rate_pct / 100))


def build_condensation_prompt(ordered_sentences, cluster_key_terms, target_words,
                               corpus_name, source_text=None):
    sentences_block = "\n".join(f"- {s['sentence']}" for s in ordered_sentences)
    terms_block = "\n".join(
        f"- {name}: " + ", ".join(terms[:15])
        for name, terms in cluster_key_terms.items() if terms
    )

    source_block = ""
    if source_text is not None:
        source_block = (
            "\n\nFor additional context, here is the full source text. Use it to "
            "faithfully represent the argument, but keep prioritizing the informative "
            "sentences and key terms above when deciding what to include:\n\n"
            f"{source_text}\n"
        )

    return (
        f'You are condensing an excerpt from the corpus "{corpus_name}" to '
        f"approximately {target_words} words (+/-5%).\n\n"
        "Below are the corpus's most informative sentences, in their original order, "
        "and the key terms of each semantic cluster identified in prior analysis. "
        "Write a single, coherent, readable condensation that:\n"
        "1. Follows the argumentative structure implied by these sentences, not "
        "necessarily their literal order.\n"
        "2. Faithfully represents the source's argument, structure, and conclusions.\n"
        f"3. Reaches approximately {target_words} words.\n"
        "4. Uses the source's own vocabulary and key terms wherever possible, "
        "especially the cluster terms listed below, but is not constrained to do so.\n"
        "5. Does not introduce examples, framings, or conclusions absent from the "
        "material below.\n\n"
        f"Informative sentences (in corpus order):\n{sentences_block}\n\n"
        f"Cluster key terms to try to preserve:\n{terms_block}"
        f"{source_block}\n\n"
        "Return only the condensed text, with no preamble or commentary."
    )


def attempt_condensation_trials(ordered_sentences, cluster_key_terms, target_words, corpus_name,
                                 model, max_trials, include_full_text, source_text=None,
                                 host=ollama_client.DEFAULT_HOST, tolerance=0.05, timeout=900,
                                 prompt_override=None):
    """Runs up to max_trials generation attempts, no input() involved.
    Returns {"text": str|None, "trials": [...], "success": bool,
    "soft_accept": bool}. The notebook decides what to do next (retry
    escalated, give up).

    If no trial lands within tolerance, falls back to the trial closest
    to target_words ("soft_accept": True) rather than discarding
    everything -- an over-length response still produces something
    usable instead of being thrown away. "success" stays False in that
    case so callers can still offer escalation; only a genuinely empty
    result (text is None) means every trial failed outright.

    `prompt_override`, if given, is used verbatim as the prompt for every
    trial instead of the one build_condensation_prompt would render --
    used by the post-condensation "regenerate" flow, where the researcher
    has freely edited the rendered default prompt (or left it untouched).

    `timeout` (seconds) is passed straight to ollama_client.generate --
    large "thinking"-capable local models can take well over the default
    600s Ollama-client timeout to respond, so this is worth raising for
    slow models rather than treating a timeout as a hard failure."""
    trials = []
    best_diff, best_text = None, None
    for trial in range(1, max_trials + 1):
        prompt = prompt_override if prompt_override is not None else build_condensation_prompt(
            ordered_sentences, cluster_key_terms, target_words, corpus_name,
            source_text=source_text if include_full_text else None,
        )
        num_ctx = estimate_num_ctx(prompt, target_words)
        text = ollama_client.generate(model, prompt, host=host, timeout=timeout,
                                       options={"num_ctx": num_ctx}, think=False)
        word_count = count_words(text)
        diff = abs(word_count - target_words)
        within_tolerance = diff <= tolerance * target_words

        trials.append({
            "trial": trial,
            "word_count": word_count,
            "target_words": target_words,
            "within_tolerance": within_tolerance,
        })

        if within_tolerance:
            return {"text": text, "trials": trials, "success": True, "soft_accept": False}

        if best_diff is None or diff < best_diff:
            best_diff, best_text = diff, text

    return {"text": best_text, "trials": trials, "success": False, "soft_accept": best_text is not None}


def run_condensation_setup(decisions, host=ollama_client.DEFAULT_HOST):
    """Interactive condensation setup: Ollama availability check plus the
    rate(s)/model/max-trials prompts. Every choice is appended to the
    decisions log via `decisions.record(...)` for later audit -- it does
    not change the interactive flow. Shared by phase2.ipynb and
    run_pipeline.py so there's one implementation, not two. Returns
    (rates, ollama_model, max_trials)."""
    if not ollama_client.is_available(host=host):
        print(
            f"Ollama does not appear to be running at {host}.\n"
            "Install it from https://ollama.com, pull a model (e.g. `ollama pull llama3`), "
            "and make sure `ollama serve` is running, then re-run this cell."
        )
    else:
        print("Ollama is available. Installed models:")
        for name in ollama_client.list_models(host=host):
            print(f"  - {name}")

    rate_input = input(
        "\nWhat condensation rate(s) would you like? (5-30%, comma-separated for multiple, e.g. '10,20'): "
    ).strip()
    rates = [int(r.strip()) for r in rate_input.split(",") if r.strip()]

    decisions.record(
        step="condensation_setup", decision_type="rate_selection",
        prompt="What condensation rate(s) would you like? (5-30%)",
        choice=rate_input, extra={"parsed_rates": rates},
    )

    ollama_model = input("\nWhich Ollama model would you like to use? ").strip()

    decisions.record(
        step="condensation_setup", decision_type="ollama_model_choice",
        prompt="Which Ollama model would you like to use?", choice=ollama_model,
    )

    max_trials = int(input("\nHow many generation trials before asking to escalate? ").strip())

    decisions.record(
        step="condensation_setup", decision_type="max_trials_choice",
        prompt="How many generation trials before asking to escalate?", choice=str(max_trials),
    )

    print(
        f"\nWill generate condensations at {rates}% using '{ollama_model}', "
        f"up to {max_trials} trial(s) each before escalation."
    )

    return rates, ollama_model, max_trials


def run_generate_for_rate(rate, ordered_sentences, cluster_key_terms, target_words, corpus_name,
                           text, model, max_trials, decisions, host=ollama_client.DEFAULT_HOST):
    """Runs the full interactive generate-then-maybe-escalate flow for one
    rate: non-escalated trials, then (if none land within tolerance) asks
    whether to retry with the full source text included, logging every
    trial and the escalation choice via `decisions.record(...)`. Saving
    the resulting text to disk stays with the caller. Shared by
    phase2.ipynb and run_pipeline.py so there's one implementation, not
    two. Returns the same {"text", "trials", "success"} shape as
    attempt_condensation_trials."""
    print(f"\n=== {rate}% condensation (target ~{target_words} words) ===")

    result = attempt_condensation_trials(
        ordered_sentences, cluster_key_terms, target_words, corpus_name,
        model=model, max_trials=max_trials, include_full_text=False, host=host,
    )

    for t in result["trials"]:
        print(
            f"  trial {t['trial']}: {t['word_count']} words (target {t['target_words']}, "
            f"{'OK' if t['within_tolerance'] else 'outside tolerance'})"
        )
        decisions.record(
            step="condensation_generation", decision_type="trial_result",
            prompt="Condensation generation trial",
            extra={"rate": rate, **t},
        )

    if not result["success"]:

        escalate = input(
            f"\n{max_trials} trial(s) at {rate}% did not hit the target word count.\n"
            "Retry with the full source text included in the prompt? (y/n): "
        ).strip().lower()

        decisions.record(
            step="condensation_generation", decision_type="escalation_prompt",
            prompt="Retry with the full source text included in the prompt?",
            options=["y", "n"], choice=escalate, extra={"rate": rate},
        )

        if escalate == "y":

            result = attempt_condensation_trials(
                ordered_sentences, cluster_key_terms, target_words, corpus_name,
                model=model, max_trials=max_trials, include_full_text=True,
                source_text=text, host=host,
            )

            for t in result["trials"]:
                print(
                    f"  [escalated] trial {t['trial']}: {t['word_count']} words "
                    f"(target {t['target_words']}, {'OK' if t['within_tolerance'] else 'outside tolerance'})"
                )
                decisions.record(
                    step="condensation_generation", decision_type="trial_result",
                    prompt="Condensation generation trial (escalated)",
                    extra={"rate": rate, "escalated": True, **t},
                )

    if result["text"] is None:
        print(f"\nGiving up on {rate}% -- no condensation within tolerance. Skipping verification/export for this rate.")
    elif result.get("soft_accept"):
        print(
            f"\n{rate}%: no trial hit the target word count within tolerance -- "
            f"using the closest trial ({count_words(result['text'])} words vs. target {target_words}) "
            "instead of discarding it."
        )

    return result


# ============================================================
# VERBATIM-OVERLAP SCAN
# ============================================================

def in_source(chunk_words, source_lower):
    phrase = " ".join(w.lower() for w in chunk_words)
    phrase_clean = re.sub(r"[^\w\s]", "", phrase)
    src_clean = re.sub(r"[^\w\s]", "", source_lower)
    return phrase_clean in src_clean


def scan_verbatim_overlap(condensed_text, source_text, min_window=4, max_window=40):
    """For each position in the condensed text, finds the longest run of
    consecutive tokens (capped at max_window) that appears verbatim in the
    source. Returns (matched_token_count, total_token_count, matched_runs)
    -- matched_runs is a list of (start_idx, end_idx, tokens)."""
    source_lower = source_text.lower()
    cond_tokens = re.findall(r"[A-Za-z']+", condensed_text)

    i, matched_runs = 0, []
    while i < len(cond_tokens):
        best_len = 0
        for length in range(min(max_window, len(cond_tokens) - i), min_window - 1, -1):
            if in_source(cond_tokens[i:i + length], source_lower):
                best_len = length
                break
        if best_len >= min_window:
            matched_runs.append((i, i + best_len, cond_tokens[i:i + best_len]))
            i += best_len
        else:
            i += 1

    matched_token_count = sum(end - start for start, end, _ in matched_runs)
    return matched_token_count, len(cond_tokens), matched_runs


# ============================================================
# INJECTION CLASSIFICATION (F/T/R/C)
# ============================================================

def _content_words(text, extra_function_words=frozenset()):
    words = text.split()
    ignore = FUNCTION_WORD_SET | extra_function_words
    return [
        w for w in words
        if re.sub(r"[^a-zA-Z]", "", w).lower() not in ignore
        and re.sub(r"[^a-zA-Z]", "", w) != ""
    ]


def is_metalinguistic(sentence):
    lowered = sentence.lower()
    return any(marker in lowered for marker in METALINGUISTIC_MARKERS)


# F ("purely connective") is decided from spaCy POS/dependency tags rather
# than a fixed word list. CCONJ/SCONJ cover true conjunctions (and/but/or/
# because/although/since/while/that/though). dep_ == "advmod" catches
# sentence-adverb discourse connectives (however/furthermore/therefore/
# thus/moreover) when they attach to a governing verb elsewhere in the
# span -- but in a short standalone fragment with no verb to attach to
# (e.g. "Furthermore," or "But then," on their own), spaCy's parser makes
# the adverb the span's ROOT instead, so pos_ == "ADV" and dep_ == "ROOT"
# is also accepted. spaCy has no dedicated "discourse adverb" tag, so both
# checks reuse dependency labels ordinary manner/degree adverbs can get
# too ("quickly", "very"); accepted as an approximation since it's only
# ever evaluated on <=4-token spans, where a stray manner/degree adverb is
# unlikely. ADP (prepositions) is deliberately excluded: "in addition"/"by
# contrast" have a real content noun as their object, so they're not
# purely connective -- they fall through to the T safety net below instead.
CONNECTIVE_POS = {"CCONJ", "SCONJ"}


def _is_connective_token(token):
    if token.pos_ in CONNECTIVE_POS:
        return True
    if token.dep_ == "advmod":
        return True
    if token.pos_ == "ADV" and token.dep_ == "ROOT":
        return True
    return False


def _non_trivial_tokens(span):
    return [t for t in span if not t.is_punct and not t.is_space]


def classify_span(sent, source_sentences, source_lower, span_id):
    """sent: a spaCy Span for one sentence (as yielded by nlp(text).sents).
    Returns a span dict, or None if the sentence is verbatim in the source
    (the taxonomy only classifies non-verbatim spans)."""
    sentence = sent.text.strip()
    words = sentence.split()

    if in_source(words, source_lower):
        return None

    tokens = _non_trivial_tokens(sent)

    if len(tokens) <= 4 and all(_is_connective_token(t) for t in tokens):
        return {
            "span_id": span_id, "type": "F", "text": sentence,
            "source_refs": [], "justification": "", "basis": "pure_connective",
        }

    if is_metalinguistic(sentence):
        return {
            "span_id": span_id, "type": "T", "text": sentence,
            "source_refs": [], "justification": "", "basis": "phrase_match",
        }

    # Safety net: a non-metalinguistic span this short isn't a reliable
    # candidate for the content-word-overlap R/C decision below (one or
    # two content words can swing that ratio from 0% to 100%), so it's
    # classified T regardless of whether it happens to contain real
    # content words -- this is what "complements" F now that F itself
    # only catches pure connective runs.
    if len(tokens) <= 6:
        return {
            "span_id": span_id, "type": "T", "text": sentence,
            "source_refs": [], "justification": "", "basis": "short_span",
        }

    span_content = {w.lower() for w in _content_words(sentence)}
    if not span_content:
        return {
            "span_id": span_id, "type": "F", "text": sentence,
            "source_refs": [], "justification": "", "basis": "zero_content_fallback",
        }

    best_idx, best_overlap, second_overlap = None, 0.0, 0.0
    for idx, src_sentence in enumerate(source_sentences):
        src_content = {w.lower() for w in _content_words(src_sentence)}
        if not src_content:
            continue
        overlap = len(span_content & src_content) / len(span_content)
        if overlap > best_overlap:
            second_overlap = best_overlap
            best_overlap, best_idx = overlap, idx
        elif overlap > second_overlap:
            second_overlap = overlap

    # A single dominant match -> paraphrase of one sentence (R).
    # Otherwise -> content drawn from/merged across multiple sentences (C).
    if best_idx is not None and best_overlap >= 0.5 and (best_overlap - second_overlap) >= 0.15:
        return {
            "span_id": span_id, "type": "R", "text": sentence,
            "source_refs": [best_idx],
            "justification": f"Paraphrases source sentence {best_idx} (content-word overlap {best_overlap:.0%}).",
        }

    return {
        "span_id": span_id, "type": "C", "text": sentence,
        "source_refs": [best_idx] if best_idx is not None else [],
        "justification": "Content does not trace cleanly to a single source sentence; treated as a compression of multiple.",
    }


def classify_condensation(condensed_text, source_text, nlp):
    """Classifies every non-verbatim sentence in the condensed text's body
    prose (p/defn blocks only -- headings and the title/author lines are
    structural, not argumentative content, and are skipped).

    Sentences are segmented block-by-block using the same
    `parse_condensed_blocks` the HTML renderer uses, rather than
    sentence-segmenting the raw text as a whole -- this guarantees every
    classified span's text is a substring of the exact block it will later
    be rendered into (render_spans matches spans to blocks by substring),
    and avoids spaCy merging an unpunctuated title line into the first
    body sentence.
    """
    source_lower = source_text.lower()
    source_sentences = [s.text.strip() for s in nlp(source_text).sents if s.text.strip()]

    all_spans = []
    span_counter = 0
    for block_type, text in parse_condensed_blocks(condensed_text):
        if block_type not in ("p", "defn"):
            continue
        for sent in nlp(text).sents:
            sentence = sent.text.strip()
            if not sentence:
                continue
            span_counter += 1
            span = classify_span(sent, source_sentences, source_lower, span_id=f"s{span_counter}")
            if span is not None:
                all_spans.append(span)

    return all_spans, source_sentences


def flag_borderline_classifications(all_spans):
    """Flags F/T spans that exceed their own type's stated definition, for
    researcher review -- not a confirmed error, a required disclosure item.
    Default is to keep the original classification unless the researcher
    requests reclassification (handled interactively in the notebook).

    classify_span's F decision is POS/dependency-based now, not lexical --
    the F check here re-runs the old word-list heuristic as an independent
    cross-check and flags disagreement, rather than re-deriving a fixed
    density threshold. T's new "basis": "short_span" spans (the safety net
    that complements F) are *expected* to contain real content words by
    design, so the density check below only applies to "basis":
    "phrase_match" spans (the original metalinguistic-phrase T rule)."""
    flags = []
    for s in all_spans:
        if s["type"] not in ("F", "T"):
            continue

        words = s["text"].split()
        content_words = _content_words(s["text"], DISCOURSE_MARKER_EXTRAS)

        if s["type"] == "F" and (len(words) > 4 or len(content_words) > 0):
            flags.append({
                "span_id": s["span_id"], "type": "F", "text": s["text"],
                "reason": (f"{len(words)} words, {len(content_words)} lexically content-bearing "
                           "token(s) -- the POS/dependency classifier called this a pure "
                           "connective/discourse-adverb fragment (<=4 tokens); the word-list "
                           "heuristic disagrees, worth a manual check."),
            })
        elif s["type"] == "T" and s.get("basis") == "phrase_match" and len(content_words) > 4:
            flags.append({
                "span_id": s["span_id"], "type": "T", "text": s["text"],
                "reason": (f"{len(content_words)} content-bearing tokens -- unusually high for a "
                           "metalinguistic span; may carry object-level content that belongs in R or C instead"),
            })
    return flags


def run_injection_review(all_spans, borderline_flags, rate, decisions):
    """Interactive borderline-classification review: prints each flagged
    span and lets the researcher keep or reclassify it. Mutates all_spans
    entries in place. Every choice is appended to the decisions log via
    `decisions.record(...)` for later audit -- it does not change the
    interactive flow. Shared by phase2.ipynb and run_pipeline.py so
    there's one implementation, not two."""
    if not borderline_flags:
        print("No borderline classifications flagged.")
        return

    print(f"\n{len(borderline_flags)} borderline classification(s) flagged for review:")

    spans_by_id = {s["span_id"]: s for s in all_spans}

    for flag in borderline_flags:

        print(f"\n  {flag['span_id']} ({flag['type']}): {flag['reason']}")
        print(f'    "{flag["text"]}"')

        new_type = input(
            "  Keep as classified, or reclassify? [ENTER = keep / F / T / R / C]: "
        ).strip().upper()

        decisions.record(
            step="injection_review", decision_type="borderline_reclassification",
            prompt="Keep as classified, or reclassify?",
            options=["", "F", "T", "R", "C"], choice=new_type,
            extra={
                "rate": rate, "span_id": flag["span_id"],
                "original_type": flag["type"],
                "new_type": new_type if new_type in ("F", "T", "R", "C") else flag["type"],
            },
        )

        if new_type in ("F", "T", "R", "C"):
            spans_by_id[flag["span_id"]]["type"] = new_type


# ============================================================
# CLUSTER COVERAGE
# ============================================================

def compute_injection_stats(all_spans, condensed_text, source_text):
    """Two independent sanity-check statistics, reported side by side
    (not one in place of the other -- they measure different things):
    non_injected_pct is the word share NOT covered by any classified F/T/R/C
    span; verbatim_overlap_pct is a mechanical token-alignment scan against
    the source, independent of the classification above."""
    injected_words = sum(len(s["text"].split()) for s in all_spans)
    condensed_words = count_words(condensed_text)
    non_injected_pct = round(100 * (1 - injected_words / condensed_words), 1) if condensed_words else 0.0

    verbatim_matched, verbatim_total, _ = scan_verbatim_overlap(condensed_text, source_text)
    verbatim_overlap_pct = round(100 * verbatim_matched / verbatim_total, 1) if verbatim_total else 0.0

    return {
        "non_injected_pct": non_injected_pct,
        "verbatim_overlap_pct": verbatim_overlap_pct,
        "verbatim_matched_tokens": verbatim_matched,
        "verbatim_total_tokens": verbatim_total,
    }


def compute_cluster_coverage(phase1_state, condensed_text, stemmer, target_pct=50.0):
    """Per Phase 1 cluster: % of its stems+n-grams present in the
    condensed text, vs. target_pct. Not derived from the (unprovided)
    source spec -- a new, documented design: OK if actual >= target, WARN
    if actual >= half the target, DARK otherwise."""
    condensed_tokens = re.findall(r"[A-Za-z']+", condensed_text)
    condensed_stems = {stemmer.stem(w.lower()) for w in condensed_tokens}
    condensed_lower = condensed_text.lower()

    report = {}
    for cluster in phase1_state.get("clusterDefs", []):
        key_terms = list(cluster.get("stems", [])) + list(cluster.get("ngrams", []))
        if not key_terms:
            continue

        present = 0
        for term in key_terms:
            normalized = term.rstrip("*").lower()
            hit = (normalized in condensed_lower) if " " in normalized else (stemmer.stem(normalized) in condensed_stems)
            if hit:
                present += 1

        actual_pct = round(100 * present / len(key_terms), 1)
        delta = round(actual_pct - target_pct, 1)
        if actual_pct >= target_pct:
            status = "OK"
        elif actual_pct >= target_pct / 2:
            status = "WARN"
        else:
            status = "DARK"

        report[cluster["name"]] = {
            "target": target_pct, "actual": actual_pct, "delta": delta, "status": status,
        }
    return report
