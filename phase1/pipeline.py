"""PEEL Phase 1 pipeline: stem extraction, GlossBERT word-sense
disambiguation, and Sentence-BERT/HDBSCAN semantic clustering.

Interactive review (accepting/editing flagged terms and clusters) stays in
phase1.ipynb, since that's genuinely interactive UX. Everything else --
stem extraction, WSD, clustering, n-gram mining, state/HTML export -- lives
here as plain functions so the notebook cells stay short and the same
logic isn't duplicated across cells.
"""

import json
import math
import re
from collections import Counter

import hdbscan
import numpy as np
import torch
from nltk.corpus import wordnet as wn
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
from transformers import BertForSequenceClassification, BertTokenizer

POS_MAP = {
    "NOUN": wn.NOUN,
    "PROPN": wn.NOUN,
    "VERB": wn.VERB,
    "ADJ": wn.ADJ,
    "ADV": wn.ADV,
}

DEFAULT_GLOSSBERT_MODEL = "jvomiranda/GlossBERT_Checkpoint"

# One prompt per Phase 1 config parameter, shared by all three interfaces
# (CLI, notebook, webapp) via log_phase1_config below -- so the exact same
# wording is logged regardless of which interface produced the run,
# instead of three independently-drifting copies of the same text.
PHASE1_CONFIG_PROMPTS = {
    "frequency_percentile": "Percentile cutoff on the frequency-ranked stem vocabulary",
    "max_stems": "Maximum number of stems to return",
    "word_frequency_percentile": "Percentile cutoff on each stem's own derived-word frequency ranking",
    "max_sentences_per_stem": "Sentences sampled per stem for WSD scoring",
    "max_synsets": "Ceiling on WordNet senses considered per word",
    "merge_duplicate_word_occurrences": "Merge all occurrences of the same word before WSD scoring?",
    "max_cluster_size": "Cluster size above which it gets automatically re-split",
    "min_clusters": "HDBSCAN min_cluster_size for the initial clustering pass",
    "min_cluster_len": "Minimum stems for a rerun/noise cluster to be kept valid",
    "glossbert_model": "GlossBERT model checkpoint to load",
    "lang_model": "spaCy language model to load",
    "sentence_embedder": "Sentence-BERT model for stem embeddings",
}


def log_phase1_config(decisions, config: dict, source: str) -> None:
    """Logs every already-resolved Phase 1 config value as its own
    decision under step="phase1_setup" (one entry per parameter, matching
    the pre-existing merge_duplicate_words_choice/
    word_frequency_percentile_choice convention) -- so the full
    configuration a run actually used is reconstructable from the
    decision log alone, not just the interactive review choices made
    afterward. `config` keys must be a subset of PHASE1_CONFIG_PROMPTS;
    call once per interface with that interface's own resolved values
    (frequency_percentile/max_stems should already reflect the sweep's
    pick, if the sweep ran)."""
    for name, value in config.items():
        decisions.record(
            step="phase1_setup", decision_type=f"{name}_choice",
            prompt=PHASE1_CONFIG_PROMPTS[name], choice=str(value),
            extra={"source": source},
        )

TABLEAU20 = [
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
    "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC",
    "#499894", "#A0CBE8", "#FFBE7D", "#FF9D9A", "#86BCB6",
    "#8CD17D", "#F1CE63", "#D4A6C8", "#FABFD2", "#D7B5A6",
]


# ============================================================
# GLOSSBERT MODEL
# ============================================================

def load_glossbert(model_id: str = DEFAULT_GLOSSBERT_MODEL, device="cpu"):
    """Loads GlossBERT from the Hugging Face Hub (cached locally after the
    first call) -- no manual checkpoint download/placement needed."""
    tokenizer = BertTokenizer.from_pretrained(model_id)
    model = BertForSequenceClassification.from_pretrained(model_id)
    model.to(device)
    model.eval()
    return tokenizer, model


# GlossBERT's tokenizer call uses max_length=128 for the (sentence, gloss)
# pair. Left as-is, HF's default pair-truncation strategy trims from the
# END of whichever sequence is longer -- for a long sentence, that means
# the tail gets cut regardless of where the marked target word actually
# sits, silently dropping it (and its quote-marking) from GlossBERT's
# input entirely whenever it occurs late in a long sentence. Verified on
# real corpus data (see docs/ai_prompts_catalog.md): in one 114-word
# Boisseau sentence, 2 of 4 flagged words were judged with zero local
# context because of exactly this.
#
# Fix: instead of feeding the whole sentence, feed a WORD-WINDOW centered
# on the target word, sized so it's guaranteed to fit within max_length
# alongside the gloss -- regardless of where in the original sentence the
# word sits. For a sentence already short enough to fit whole, the window
# request is >= the sentence's own length, so this is a no-op (byte-
# identical to the old behavior) for the common case; it only changes
# anything for the sentences that were silently losing the word before.
_GLOSSBERT_MAX_LENGTH = 128
# glossbert_predict_merged concatenates multiple occurrences into one
# paragraph before scoring, so it gets a much larger budget than a single
# occurrence needs -- GlossBERT is BERT-base underneath (max_position_
# embeddings=512), so 512 is the model's actual ceiling, not an arbitrary
# increase. Single-occurrence glossbert_predict deliberately stays at 128:
# one sentence window rarely needs more, and keeping it small keeps that
# path's latency down since it runs once per occurrence rather than once
# per merged group.
_GLOSSBERT_MERGED_MAX_LENGTH = 512
_GLOSSBERT_SPECIAL_TOKENS = 3            # [CLS] + [SEP] + [SEP]
_GLOSSBERT_QUOTE_TOKENS_PER_MARK = 2     # the two literal '"' characters around one marked word
_GLOSSBERT_TOKENS_PER_WORD = 1.6         # measured p95 (not mean 1.3) tokens/word on real corpus
                                           # text with this tokenizer -- deliberately pessimistic
                                           # so the WORD-based window still fits after wordpiece
                                           # subword splitting, not just on average


def _glossbert_words_budget(gloss, tokenizer, n_marks=1, min_words_each_side=1, max_length=_GLOSSBERT_MAX_LENGTH):
    """How many words of surrounding context (each side, per marked
    occurrence) fit in `max_length` once `gloss` and n_marks sets of
    quote-marking are accounted for. n_marks=1 for a single occurrence;
    >1 when glossbert_predict_merged shares one budget (max_length=512,
    see _GLOSSBERT_MERGED_MAX_LENGTH) across several concatenated
    occurrences."""
    # truncation=True/max_length here is purely defensive -- a WordNet
    # gloss is always short, this measurement realistically never gets
    # anywhere near max_length -- but it silences transformers' "token
    # indices sequence length" warning for the (currently theoretical)
    # case of an unusually long gloss, with no effect on the result: a
    # measurement this method would ever act on is already far below
    # max_length, so capping it there loses no decision-relevant
    # precision (mirrors the identical, non-theoretical fix in
    # _fit_contexts_to_budget's n_tokens(), which real Boisseau sentences
    # DO trigger).
    gloss_tokens = len(tokenizer(gloss, truncation=True, max_length=max_length)["input_ids"]) - 2
    quote_tokens = _GLOSSBERT_QUOTE_TOKENS_PER_MARK * n_marks
    budget = max_length - _GLOSSBERT_SPECIAL_TOKENS - quote_tokens - gloss_tokens
    budget = max(budget, min_words_each_side * 2 * n_marks)
    per_mark_budget = budget / n_marks
    return max(min_words_each_side, int(per_mark_budget / _GLOSSBERT_TOKENS_PER_WORD / 2))


def _window_around_match(sentence, match_start, match_end, max_words_each_side):
    """Word-based window around sentence[match_start:match_end], extended
    up to max_words_each_side words on each side (clamped at sentence
    boundaries; '...' marks an actual cut).

    Strips literal straight double-quotes (") from the window text before
    the caller wraps the matched word in its own pair of them -- GlossBERT
    (per its paper) uses "..." as the ONLY signal for which word is being
    disambiguated, so if the surrounding context already contains a
    straight-quote pair (a direct quotation, scare-quotes, code-like
    text), the model would see multiple indistinguishable quote-marked
    spans with no way to tell which one is the real target. Verified this
    doesn't currently affect Boisseau (0 literal straight quotes in that
    corpus; it uses typographic curly quotes exclusively, which tokenize
    to entirely different token ids than the marking character -- no
    collision), but a different corpus could easily contain them.
    Replaced with a space, not deleted outright, so two words separated
    only by a quote don't get accidentally fused together."""
    before_words = sentence[:match_start].replace('"', " ").split()
    after_words = sentence[match_end:].replace('"', " ").split()

    kept_before = before_words[-max_words_each_side:] if max_words_each_side else []
    kept_after = after_words[:max_words_each_side] if max_words_each_side else []

    prefix = "... " if len(kept_before) < len(before_words) else ""
    suffix = " ..." if len(kept_after) < len(after_words) else ""
    return prefix + " ".join(kept_before), " ".join(kept_after) + suffix


def glossbert_predict(occurrence, tokenizer, model, device, pos_map=POS_MAP, max_synsets=5):
    word = occurrence["word"]
    pos = occurrence["pos"]
    sentence = occurrence["sentence"]
    wn_pos = pos_map.get(pos)

    synsets = wn.synsets(word, pos=wn_pos)
    if not synsets:
        return None
    synsets = synsets[:max_synsets]

    # Case-insensitive: word is normalized lowercase (see map_stems_to_sentences),
    # but sentence keeps its original casing (e.g. an all-caps heading) --
    # a case-sensitive .replace() would silently fail to find the word there.
    # \b-anchored: without word boundaries, a short word like "ai" matches
    # the substring inside "main"/"again"/"certain"/"maintain" etc. --
    # re.search returns the FIRST such match in the sentence, so a stem
    # whose letters happen to occur inside an earlier, unrelated word gets
    # windowed and marked around that unrelated word instead of its real
    # target. Verified on real Boisseau data: "ai" was being marked inside
    # "main" while the real "AI" later in the same sentence was ignored.
    match = re.search(r"\b" + re.escape(word) + r"\b", sentence, re.IGNORECASE)
    if match:
        # Sized once, using the LONGEST candidate gloss, so every synset
        # below is scored against the identical sentence window -- only
        # the gloss differs, keeping the cross-sense comparison fair
        # (a synset with a longer gloss doesn't get a smaller window than
        # its competitors).
        worst_case_gloss = max((syn.definition() for syn in synsets), key=len)
        max_words_each_side = _glossbert_words_budget(worst_case_gloss, tokenizer)
        before, after = _window_around_match(sentence, match.start(), match.end(), max_words_each_side)
        marked_sentence = f'{before} "{match.group()}" {after}'.strip()
    else:
        marked_sentence = sentence

    results = []
    for syn in synsets:
        gloss = syn.definition()
        encoding = tokenizer(
            marked_sentence, gloss,
            return_tensors="pt", truncation=True,
            max_length=128, padding="max_length",
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)
            match_score = probs[0][1].item()

        results.append({"synset": syn, "definition": gloss, "score": match_score})

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def glossbert_predict_merged(word, occurrences, tokenizer, model, device, pos_map=POS_MAP, max_synsets=5):
    """Merges every occurrence's sentence (each independently marked around
    its own instance of `word` before merging, so this doesn't inherit
    glossbert_predict's "only the first regex match" limitation) into one
    paragraph and scores it against WordNet candidate synsets exactly
    once -- used when merge_duplicate_word_occurrences is enabled, so a
    word occurring multiple times in the sampled window yields at most
    one prediction instead of one per occurrence (see
    run_glossbert_analysis).

    Uses a 512-token budget (_GLOSSBERT_MERGED_MAX_LENGTH) rather than
    glossbert_predict's 128 -- GlossBERT is BERT-base underneath, so 512 is
    its actual position-embedding ceiling, and a merged paragraph has much
    more text to fit than a single occurrence does. The merge window is
    bounded by max_sentences_per_stem, not a separate uncapped setting.
    Each occurrence's own sentence is windowed around its match the same
    way glossbert_predict windows a single occurrence (see
    _window_around_match), with the shared 512-token budget split N ways
    across all occurrences being merged -- so every occurrence keeps at
    least some surrounding context and its quote-marking, rather than
    later occurrences in the join order being silently dropped by tail-
    truncation once the paragraph runs long."""
    pos = occurrences[0]["pos"]
    wn_pos = pos_map.get(pos)

    synsets = wn.synsets(word, pos=wn_pos)
    if not synsets:
        return None
    synsets = synsets[:max_synsets]

    worst_case_gloss = max((syn.definition() for syn in synsets), key=len)
    max_words_each_side = _glossbert_words_budget(
        worst_case_gloss, tokenizer, n_marks=len(occurrences), max_length=_GLOSSBERT_MERGED_MAX_LENGTH,
    )

    marked_sentences = []
    for occurrence in occurrences:
        sentence = occurrence["sentence"]
        # \b-anchored -- see glossbert_predict's identical fix for why an
        # unanchored substring search can mark the wrong word entirely.
        match = re.search(r"\b" + re.escape(word) + r"\b", sentence, re.IGNORECASE)
        if match:
            before, after = _window_around_match(sentence, match.start(), match.end(), max_words_each_side)
            marked_sentences.append(f'{before} "{match.group()}" {after}'.strip())
        else:
            marked_sentences.append(sentence)
    merged_text = " ".join(marked_sentences)

    results = []
    for syn in synsets:
        gloss = syn.definition()
        encoding = tokenizer(
            merged_text, gloss,
            return_tensors="pt", truncation=True,
            max_length=_GLOSSBERT_MERGED_MAX_LENGTH, padding="max_length",
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)
            match_score = probs[0][1].item()

        results.append({"synset": syn, "definition": gloss, "score": match_score})

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


# ============================================================
# STEM EXTRACTION & MAPPING
# ============================================================

def extract_top_stems(doc, stemmer, frequency_percentile, max_stems) -> dict:
    """`frequency_percentile` is a statistical percentile CUTOFF on the
    stem-frequency ranking (0-1): only stems at or above that percentile
    of the ranking are kept, so a HIGHER value is a stricter bar and
    selects FEWER (the most frequent) stems -- e.g. 0.75 keeps only the
    top 25% most-frequent stems, matching the everyday sense of "90th
    percentile cutoff" meaning "top 10%". Internally this is
    `kept_fraction = 1 - frequency_percentile` of the ranked vocabulary,
    since the true value-based percentile of raw frequency counts
    degenerates on real (Zipfian) text -- e.g. on Boisseau, 43.6% of
    stems occur exactly once, so the 25th percentile of frequency VALUES
    is already 1, and "keep everything >= 1" selects 100% of the
    vocabulary instead of 75%. Rank-based selection sidesteps that tie
    pileup and gives a predictable stem count regardless of how skewed
    the frequency distribution is."""
    tokens = [
        stemmer.stem(token.text.lower())
        for token in doc
        if not token.is_stop and not token.is_punct and not token.is_space and token.is_alpha
    ]

    sorted_freq = sorted(Counter(tokens).items(), key=lambda x: x[1], reverse=True)
    kept_fraction = 1 - frequency_percentile
    final_n = min(math.ceil(len(sorted_freq) * kept_fraction), max_stems)
    return dict(sorted_freq[:final_n])


def describe_stem_cap(top_stems, doc, stemmer, frequency_percentile, max_stems) -> str | None:
    """One-line note for when max_stems -- not frequency_percentile --
    actually determined the final stem count, since a LOWER
    frequency_percentile keeps a LARGER fraction of the ranked vocabulary
    (it's easy to pick a low frequency_percentile, still exceed
    max_stems, and be surprised the count didn't grow further). Returns
    None when the cap didn't bind (frequency_percentile alone already
    produced <= max_stems)."""
    if len(top_stems) < max_stems:
        return None
    vocab_size = len({
        stemmer.stem(token.text.lower())
        for token in doc
        if not token.is_stop and not token.is_punct and not token.is_space and token.is_alpha
    })
    kept_fraction = 1 - frequency_percentile
    uncapped = math.ceil(vocab_size * kept_fraction)
    if uncapped <= max_stems:
        return None
    return (
        f"(frequency_percentile={frequency_percentile} alone would select {uncapped} stems "
        f"from this corpus's {vocab_size}-stem vocabulary -- max_stems={max_stems} "
        "capped it; a lower frequency_percentile would not have selected fewer)"
    )


def map_stems_to_sentences(doc, top_stems, stemmer) -> dict:
    stem_occurrences = {}
    for sent in tqdm(list(doc.sents), desc="Sentence mapping"):
        sent_text = sent.text.strip()
        for token in sent:
            if not token.is_alpha:
                continue
            stem = stemmer.stem(token.text.lower())
            if stem not in top_stems:
                continue
            stem_occurrences.setdefault(stem, []).append({
                # Lowercased so e.g. the same word occurring once in an
                # all-caps heading and once in normal prose is treated as
                # one consistent form downstream (display, dedup, cluster
                # naming), not two -- see glossbert_predict's
                # case-insensitive marking, which is what keeps this safe.
                "word": token.text.lower(),
                "stem": stem,
                "pos": token.pos_,
                "sentence": sent_text,
                "start": token.idx,
                "end": token.idx + len(token.text),
            })
    return stem_occurrences


def filter_stem_occurrences_by_word_frequency(stem_occurrences, word_frequency_percentile, min_words_per_stem=1) -> dict:
    """Trims each stem's DERIVED WORDS (distinct surface forms, e.g.
    "observation"/"observes"/"observed" all stemming to "observ") down to
    only those at or above a percentile cutoff of THAT STEM'S OWN
    word-frequency ranking -- same rank-based percentile-cutoff semantics
    as extract_top_stems's frequency_percentile (higher = fewer, more
    frequent kept), just applied one level down: from stems to the words
    derived from each stem, instead of pooling every stem's words into one
    global ranking (which would let a single low-frequency stem lose ALL
    of its words while a high-frequency stem barely loses any).

    Run this right after map_stems_to_sentences and before
    run_glossbert_analysis: run_glossbert_analysis's own
    max_sentences_per_stem cap samples occurrences in raw DOCUMENT ORDER,
    not by word frequency, so without this filter which derived words
    even get a chance to be sampled (and therefore reviewed) is
    essentially arbitrary. This filter makes that pool frequency-driven
    first, concentrating GlossBERT scoring and manual review on each
    stem's genuinely representative forms.

    min_words_per_stem guarantees every stem keeps at least its N
    most-frequent derived words, so a stem is never emptied out entirely
    by the cutoff. word_frequency_percentile=0 (the default in every
    interface) keeps every derived word -- byte-identical to no
    filtering."""
    if word_frequency_percentile <= 0:
        return stem_occurrences

    kept_fraction = 1 - word_frequency_percentile
    filtered = {}
    for stem, occurrences in stem_occurrences.items():
        word_counts = Counter(occ["word"] for occ in occurrences)
        ranked_words = [w for w, _ in sorted(word_counts.items(), key=lambda x: x[1], reverse=True)]
        n_keep = max(min_words_per_stem, math.ceil(len(ranked_words) * kept_fraction))
        kept_words = set(ranked_words[:n_keep])
        filtered[stem] = [occ for occ in occurrences if occ["word"] in kept_words]
    return filtered


# ============================================================
# GLOSSBERT ANALYSIS
# ============================================================

def record_accepted_instance(accepted_definitions, stem, word, pos, definition, sentence, count):
    entry = accepted_definitions.setdefault(
        stem, {"words": set(), "pos": set(), "count": count, "instances": []}
    )
    entry["words"].add(word)
    entry["pos"].add(pos)
    entry["instances"].append({"word": word, "definition": definition, "sentence": sentence})
    return entry


def purge_word_from_occurrences(stem_occurrences, stem, word):
    """Removes every occurrence of `word` from stem_occurrences[stem] in
    place. Used when a flagged word is deleted from review: skipping
    record_accepted_instance alone keeps it out of embeddings/clustering
    (which only read accepted_definitions), but stem_occurrences itself is
    reused, unfiltered, later by cluster naming (build_stem_word_frequency_table)
    and by any large-cluster/noise resplit that pulls fresh context text
    straight from it (recluster_large_clusters, recluster_noise) -- without
    this purge a deleted word's text could silently resurface there."""
    if stem not in stem_occurrences:
        return
    stem_occurrences[stem] = [occ for occ in stem_occurrences[stem] if occ["word"] != word]


def describe_wordnet_excluded_stems(stem_occurrences, max_sentences_per_stem=5,
                                     merge_duplicate_word_occurrences=False, pos_map=POS_MAP) -> str | None:
    """One-line note listing every selected stem that run_glossbert_analysis
    will silently drop before flagged-term review ever sees it: its own
    `if not synsets: continue` skip, replicated here read-only, for every
    word in the same capped `occurrences[:max_sentences_per_stem]` sample it
    actually scores. If that leaves zero WordNet-eligible words the stem
    produces no flagged item and no accepted instance, so it just
    disappears between "Selected N top stems" and the final embedded count
    with no other trace anywhere in the pipeline's output.

    Must mirror run_glossbert_analysis's merged-vs-non-merged branch
    exactly, not just check "does any occurrence's word have synsets" --
    they differ in a way that changes the answer:
      - merge_duplicate_word_occurrences=True: WordNet POS for a word is
        taken from only its FIRST sampled occurrence
        (`word_occurrences[0]["pos"]`) -- a later occurrence of the same
        word tagged with a different, synset-having POS is never consulted.
      - False (default): each occurrence is checked under its own POS
        individually, so the same word can be eligible via one occurrence
        even if another occurrence of it isn't.
    Confirmed against a real run: with merge=True, `defer`/`et`/`non`/
    `veritist` all lose their only sampled word to this skip (e.g. `defer`'s
    first sampled occurrence is tagged NOUN, 0 synsets, even though later
    occurrences beyond the cap are tagged VERB with 2) while `ross` is
    correctly left out here since it has real synsets and was a genuine
    flagged-then-deleted case, not a silent one.

    Returns None when every selected stem has at least one WordNet-eligible
    word this way."""
    excluded = []
    for stem, all_occurrences in stem_occurrences.items():
        occurrences = all_occurrences[:max_sentences_per_stem]
        if not occurrences:
            continue

        if merge_duplicate_word_occurrences:
            by_word = {}
            for occurrence in occurrences:
                by_word.setdefault(occurrence["word"], []).append(occurrence)
            has_eligible_word = any(
                wn.synsets(word, pos=pos_map.get(word_occurrences[0]["pos"]))
                for word, word_occurrences in by_word.items()
            )
        else:
            has_eligible_word = any(
                wn.synsets(occurrence["word"], pos=pos_map.get(occurrence["pos"]))
                for occurrence in occurrences
            )

        if not has_eligible_word:
            excluded.append(stem)

    if not excluded:
        return None
    return (
        f"NOTE: {len(excluded)} selected stem(s) have no WordNet-eligible derived words and "
        f"will be silently excluded from flagged-term review and embeddings: {', '.join(sorted(excluded))}"
    )


def run_glossbert_analysis(top_stems, stem_occurrences, tokenizer, model, device,
                            pos_map=POS_MAP, max_sentences_per_stem=5, max_synsets=5,
                            merge_duplicate_word_occurrences=False):
    accepted_definitions = {}
    flagged_words = []

    for stem, count in tqdm(top_stems.items(), total=len(top_stems), desc="GlossBERT"):
        occurrences = stem_occurrences.get(stem, [])[:max_sentences_per_stem]
        if not occurrences:
            continue

        if merge_duplicate_word_occurrences:
            # Group the (already-capped) sampled occurrences by exact word
            # text, so each distinct lexical word is scored exactly once
            # against its concatenated contexts instead of once per
            # occurrence -- this is what collapses what would otherwise be
            # multiple, potentially differently-mismatched, independent
            # predictions for the same word into a single review item.
            by_word = {}
            for occurrence in occurrences:
                by_word.setdefault(occurrence["word"], []).append(occurrence)

            mismatch_found = False
            word_results = {}
            for word, word_occurrences in by_word.items():
                wn_pos = pos_map.get(word_occurrences[0]["pos"])
                synsets = wn.synsets(word, pos=wn_pos)
                if not synsets:
                    continue
                default_sense = synsets[0]

                results = glossbert_predict_merged(
                    word, word_occurrences, tokenizer, model, device, pos_map, max_synsets
                )
                if results is None:
                    continue
                best_sense = results[0]["synset"]
                word_results[word] = default_sense

                if best_sense.name() != default_sense.name():
                    mismatch_found = True
                    flagged_words.append({
                        "word": word,
                        "stem": stem,
                        "pos": word_occurrences[0]["pos"],
                        "count": count,
                        "sentence": " ".join(o["sentence"] for o in word_occurrences),
                        "default_sense": default_sense.name(),
                        "default_definition": default_sense.definition(),
                        "predicted_sense": best_sense.name(),
                        "predicted_definition": best_sense.definition(),
                        "top_candidates": [
                            {"sense": r["synset"].name(), "definition": r["definition"], "score": round(r["score"], 4)}
                            for r in results[:max_synsets]
                        ],
                    })

            if not mismatch_found:
                for occurrence in occurrences:
                    default_sense = word_results.get(occurrence["word"])
                    if default_sense is None:
                        continue
                    record_accepted_instance(
                        accepted_definitions, stem, occurrence["word"], occurrence["pos"],
                        default_sense.definition(), occurrence["sentence"], count,
                    )
            continue

        mismatch_found = False
        seen_mismatches = set()
        for occurrence in occurrences:
            word = occurrence["word"]
            wn_pos = pos_map.get(occurrence["pos"])
            synsets = wn.synsets(word, pos=wn_pos)
            if not synsets:
                continue
            default_sense = synsets[0]

            results = glossbert_predict(occurrence, tokenizer, model, device, pos_map, max_synsets)
            if results is None:
                continue
            best_sense = results[0]["synset"]

            if best_sense.name() != default_sense.name():
                mismatch_found = True
                # Two occurrences of the same word (e.g. once from an
                # all-caps heading, once from normal prose) that resolve
                # to the same default/predicted senses are the same
                # review item -- collapse them instead of flagging twice.
                dedup_key = (word, default_sense.name(), best_sense.name())
                if dedup_key in seen_mismatches:
                    continue
                seen_mismatches.add(dedup_key)
                flagged_words.append({
                    "word": word,
                    "stem": stem,
                    "pos": occurrence["pos"],
                    "count": count,
                    "sentence": occurrence["sentence"],
                    "default_sense": default_sense.name(),
                    "default_definition": default_sense.definition(),
                    "predicted_sense": best_sense.name(),
                    "predicted_definition": best_sense.definition(),
                    "top_candidates": [
                        {"sense": r["synset"].name(), "definition": r["definition"], "score": round(r["score"], 4)}
                        for r in results[:max_synsets]
                    ],
                })

        if not mismatch_found:
            for occurrence in occurrences:
                word = occurrence["word"]
                wn_pos = pos_map.get(occurrence["pos"])
                synsets = wn.synsets(word, pos=wn_pos)
                if not synsets:
                    continue
                record_accepted_instance(
                    accepted_definitions, stem, word, occurrence["pos"],
                    synsets[0].definition(), occurrence["sentence"], count,
                )

    return accepted_definitions, flagged_words


def resolve_flagged_choice(item, choice):
    """Pure resolution of a flagged-term review choice; no input() here.
    Returns (kind, value): kind is "definition" (value=chosen text),
    "manual" (caller must still prompt for the custom text),
    "accept_all", "exit", "delete" (remove the word from all following
    steps), or "invalid".

    Candidate slots are 1..len(top_candidates); the manual-entry slot is
    whatever number comes right after the last candidate -- both derived
    from the actual candidate count (driven by max_synsets) rather than
    a fixed "3 candidates + slot 4" assumption."""
    choice = choice.strip()
    if choice.lower() == "accept all":
        return "accept_all", None
    if choice.lower() == "exit":
        return "exit", None
    if choice.lower() == "delete":
        return "delete", None
    if choice == "0":
        return "definition", item["default_definition"]
    candidates = item["top_candidates"]
    if choice == str(len(candidates) + 1):
        return "manual", None
    if choice.isdigit() and 1 <= int(choice) <= len(candidates):
        return "definition", candidates[int(choice) - 1]["definition"]
    return "invalid", None


def save_glossbert_output(accepted_definitions, path):
    with open(path, "w", encoding="utf-8") as out:
        out.write("stem\twords\tcount\tpos\tdefinitions\n")
        for stem, data in accepted_definitions.items():
            out.write(f"{stem}\t{data['words']}\t{data['count']}\t{data['pos']}\t{data['instances']}\n")


def print_accepted_definitions(accepted_definitions):
    print("\n===================================")
    print("ACCEPTED DEFINITIONS")
    print("===================================\n")
    for stem, data in accepted_definitions.items():
        print({
            "words": sorted(data["words"]),
            "stem": stem,
            "pos": sorted(data["pos"]),
            "instances": data["instances"],
        })


def print_flagged_item(item):
    print("\n-----------------------------------")
    print(f"WORD: {item['word']}")
    print(f"STEM: {item['stem']}")
    print(f"POS: {item['pos']}")
    print("\nSENTENCE:")
    print(item["sentence"])
    print("\nDEFAULT SENSE:")
    print(item["default_sense"])
    print(item["default_definition"])
    print("\nPREDICTED SENSE:")
    print(item["predicted_sense"])
    print(item["predicted_definition"])
    print("\nTOP CANDIDATES:")
    for idx, candidate in enumerate(item["top_candidates"], start=1):
        print(f"[{idx}] {candidate['sense']}")
        print(f"    DEF: {candidate['definition']}")
        print(f"    SCORE: {candidate['score']}")


def print_flagged_words(flagged_words):
    print("\n===================================")
    print("FLAGGED TERMS")
    print("===================================\n")
    for item in flagged_words:
        print_flagged_item(item)


def run_flagged_term_review(flagged_words, accepted_definitions, decisions, stem_occurrences=None):
    """Interactive review of GlossBERT's flagged sense mismatches: accept
    all predictions, review one by one, or select specific terms. Mutates
    accepted_definitions in place. Every choice is appended to the
    decisions log via `decisions.record(...)` for later audit -- it does
    not change the interactive flow. Shared by phase1.ipynb and
    run_pipeline.py so there's one implementation, not two.

    stem_occurrences, if given, lets a per-item "delete" choice also purge
    the word from the raw occurrence pool (not just skip accepting it) --
    see purge_word_from_occurrences."""
    print("\n===================================")
    print("FLAGGED TERM REVIEW")
    print("===================================\n")

    if not flagged_words:
        print("No flagged terms found.")
        return

    accept_all = input(
        "\nAccept all predicted definitions from flagged terms? (y/n): "
    ).strip().lower()

    decisions.record(
        step="flagged_term_review", decision_type="review_mode",
        prompt="Accept all predicted definitions from flagged terms?",
        options=["y", "n"], choice=accept_all,
    )

    if accept_all == "y":

        for item in flagged_words:
            record_accepted_instance(
                accepted_definitions, item["stem"], item["word"], item["pos"],
                item["predicted_definition"], item["sentence"], item["count"],
            )

        print("\nAll predicted definitions accepted.")
        return

    review_mode = input(
        "\nReview flagged terms:\n"
        "[1] One by one\n"
        "[2] Select from all terms\n\n"
        "Choice: "
    ).strip()

    decisions.record(
        step="flagged_term_review", decision_type="review_submode",
        prompt="Review flagged terms mode", options=["1", "2"], choice=review_mode,
    )

    def review_flagged_item(item):

        print_flagged_item(item)

        print("\n[0] Keep default definition")
        print(f"[{len(item['top_candidates']) + 1}] Enter manual definition")
        print("Or type 'Accept all' to accept all remaining flagged terms")
        print("Or type 'Delete' to remove this word from all following steps")

        choice = input("\nChoice: ").strip()

        kind, value = resolve_flagged_choice(item, choice)

        if kind == "manual":
            value = input("\nEnter custom definition: ").strip()

        decisions.record(
            step="flagged_term_review", decision_type="per_term_choice",
            prompt="Flagged term review choice", options=item["top_candidates"],
            choice=choice,
            extra={
                "word": item["word"],
                "stem": item["stem"],
                "default_definition": item["default_definition"],
                "resulting_definition": value if kind in ("definition", "manual") else None,
            },
        )

        if kind in ("accept_all", "exit"):
            return kind

        if kind == "invalid":
            print("\nInvalid option.")
            return None

        if kind == "delete":
            if stem_occurrences is not None:
                purge_word_from_occurrences(stem_occurrences, item["stem"], item["word"])
            print("\nWord deleted from all following steps.")
            return None

        record_accepted_instance(
            accepted_definitions, item["stem"], item["word"], item["pos"],
            value, item["sentence"], item["count"],
        )

        print("\nDefinition updated.")
        return None

    if review_mode == "1":

        for idx, item in enumerate(flagged_words):

            result = review_flagged_item(item)

            if result == "accept_all":

                for remaining_item in flagged_words[idx:]:
                    record_accepted_instance(
                        accepted_definitions, remaining_item["stem"], remaining_item["word"],
                        remaining_item["pos"], remaining_item["predicted_definition"],
                        remaining_item["sentence"], remaining_item["count"],
                    )

                print("\nAll remaining flagged terms accepted.")
                break

            elif result == "exit":

                print("\nExiting review.")
                break

    elif review_mode == "2":

        print("\nFLAGGED TERMS:\n")
        for item in flagged_words:
            print(f"- {item['word']} (stem={item['stem']})")

        while True:

            selected_word = input("\nType a word to review (or 'exit'): ").strip()

            if selected_word.lower() == "exit":
                break

            found = False
            for item in flagged_words:
                if item["word"].lower() == selected_word.lower():
                    review_flagged_item(item)
                    found = True

            if not found:
                print("\nWord not found.")


# ============================================================
# SEMANTIC CLUSTERING
# ============================================================

# Every embedding text built below follows a template combining a short
# label (stem or word), an "Observed words"/context section, and a
# "Candidate senses" section, fed to all-MiniLM-L6-v2 (max_seq_length=256,
# measured directly -- not the commonly-assumed 128). "Contexts" is the
# one section whose length was previously unbounded (up to 3 full source
# sentences, verbatim, however long); on real Boisseau data this alone
# was enough to push 2 of the 3 longest stem-texts over 256 tokens,
# silently truncating away part or all of "Candidate senses" -- the
# section that comes last in the string and is therefore always what
# gets cut first. See docs/ai_prompts_catalog.md for the measured cases.
#
# Fix: rather than a fixed word-count cap derived from one corpus's
# measured statistics (which would misfit a source with different
# vocabulary -- longer compound words, denser subword-splitting, a
# non-English source, a field whose WordNet-adjacent gloss text runs
# long, etc.), _fit_contexts_to_budget measures the ACTUAL token cost of
# every other section of THIS text with the real tokenizer at call time,
# and gives the context sentences whatever budget is left out of
# max_seq_length. Each context sentence is then capped using its OWN
# real tokens-per-word ratio (also measured on the spot), not a
# corpus-wide assumed ratio -- so the cap adapts automatically to
# whatever the current source's actual tokenization looks like, and
# candidate senses (the section most worth protecting, since it's WSD's
# actual output) is never the one silently sacrificed.
_SBERT_SPECIAL_TOKENS = 2  # [CLS] + [SEP] around one single (non-pair) input


def _fit_contexts_to_budget(tokenizer, max_seq_length, fixed_parts, contexts, min_words_per_context=3):
    """Caps `contexts` (a list of sentences) so that, combined with every
    other literal chunk of the template (`fixed_parts` -- labels,
    observed words, candidate senses, punctuation), the whole text's real
    token count fits within max_seq_length. All measurements come from
    tokenizing the actual text being built, right now -- nothing here is
    a constant borrowed from a different corpus. A no-op when everything
    already fits."""
    # truncation=True/max_length here is purely defensive -- this measures
    # a RAW, not-yet-trimmed context sentence's real length, purely to
    # decide how much to cut. Verified on real Boisseau data: a long
    # academic sentence can genuinely be 295+ tokens on its own, which
    # without this would print transformers' harmless-but-alarming-sounding
    # "Token indices sequence length is longer than..." warning on every
    # such sentence -- cosmetic only (the actual embedder.encode() calls
    # this feeds into never see anything over max_seq_length regardless,
    # verified directly: 0 of 131 real stem-embedding texts and 0 of the
    # real recluster_noise texts exceeded 256 tokens on this exact
    # corpus/config). Capping the MEASUREMENT at max_seq_length changes
    # nothing decision-relevant: it's only ever compared against a
    # per-sentence `share` that's always far smaller than max_seq_length
    # itself, so "exactly at the cap" vs "the real, larger value" both
    # correctly trigger trimming.
    def n_tokens(text):
        return len(tokenizer(text, add_special_tokens=False, truncation=True, max_length=max_seq_length)["input_ids"])

    if not contexts:
        return []

    fixed_tokens = sum(n_tokens(part) for part in fixed_parts)
    budget = max_seq_length - _SBERT_SPECIAL_TOKENS - fixed_tokens
    if budget <= 0:
        print(
            f"WARNING: label + observed words + candidate senses alone use "
            f"{fixed_tokens} tokens, already at or beyond this text's "
            f"{max_seq_length}-token budget -- context sentences are being cut "
            "to a minimal fallback and the embedder's own truncation may still "
            "reach into candidate senses for this one entry."
        )
        budget = 0

    capped = []
    remaining_budget = budget
    remaining_sentences = len(contexts)
    for context in contexts:
        share = max(remaining_budget // remaining_sentences, 0)
        context_tokens = n_tokens(context)
        words = context.split()
        if context_tokens <= share:
            kept = context
        else:
            ratio = context_tokens / len(words) if words else 1.0
            max_words = max(min_words_per_context, int(share / ratio)) if ratio > 0 else len(words)
            max_words = min(max_words, len(words))
            kept = " ".join(words[:max_words])
            if max_words < len(words):
                kept += " ..."
        capped.append(kept)
        remaining_budget -= n_tokens(kept)
        remaining_sentences -= 1
    return capped


def build_stem_embeddings(accepted_definitions, embedder_name):
    """One embedding point per unique stem (an aggregated representative
    text over its instances), not one point per instance -- otherwise a
    stem with many accepted instances gets that many points in the HDBSCAN
    input and can out-vote its own density requirement."""
    embedder = SentenceTransformer(embedder_name)

    stem_names = []
    stem_texts = []
    for stem, data in accepted_definitions.items():
        instances = data["instances"]
        if not instances:
            continue

        observed_words = list(dict.fromkeys(inst["word"] for inst in instances))[:5]
        contexts = list(dict.fromkeys(inst["sentence"] for inst in instances))[:3]
        definitions = list(dict.fromkeys(inst["definition"] for inst in instances))[:5]

        prefix = f"Stem: {stem}. Observed words: " + ", ".join(observed_words) + ". Contexts: "
        suffix = ". Candidate senses: " + " ".join(definitions)
        contexts = _fit_contexts_to_budget(embedder.tokenizer, embedder.max_seq_length, [prefix, suffix], contexts)

        stem_texts.append(prefix + " ".join(contexts) + suffix)
        stem_names.append(stem)

    embeddings = embedder.encode(stem_texts, convert_to_numpy=True, normalize_embeddings=True)
    return embedder, stem_texts, stem_names, embeddings


def cluster_stem_embeddings(embeddings, stem_names, min_cluster_size, min_cluster_len=3):
    """Any cluster with fewer than min_cluster_len unique stems is
    dissolved back into noise (label -1), same threshold the recluster
    steps already enforce, so it flows into extract_noise_stems/
    recluster_noise for a second attempt or manual merging in review."""
    labels = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(embeddings)

    clusters = {}
    for stem, label in zip(stem_names, labels):
        if label == -1:
            continue
        bucket = clusters.setdefault(label, [])
        if stem not in bucket:
            bucket.append(stem)

    dissolved_labels = {label for label, stems in clusters.items() if len(set(stems)) < min_cluster_len}
    clusters = {label: stems for label, stems in clusters.items() if label not in dissolved_labels}
    labels = np.array([-1 if label in dissolved_labels else label for label in labels])

    return labels, clusters


def build_stem_word_frequency_table(stem_occurrences):
    return {
        stem: Counter(occ["word"].lower() for occ in occurrences)
        for stem, occurrences in stem_occurrences.items()
    }


def rename_clusters(cluster_dict, stem_word_frequencies):
    """Names each cluster after its single most-frequent observed word.
    Two unrelated clusters can easily share a dominant word (a corpus-wide
    high-frequency term like "trust" will top many different clusters'
    counters), so names are disambiguated with the same numeric-suffix
    strategy merge_named_clusters already uses for main-vs-noise collisions
    -- without it, a plain dict assignment here would silently overwrite an
    earlier same-named cluster and drop its stems entirely with no warning
    (verified on real Boisseau data: 6 of 20 raw noise clusters collided
    this way, all lost, before this fix)."""
    renamed = {}
    for label, stems in cluster_dict.items():
        cluster_word_counter = Counter()
        for stem in stems:
            if stem in stem_word_frequencies:
                cluster_word_counter.update(stem_word_frequencies[stem])

        if not cluster_word_counter:
            continue

        max_freq = max(cluster_word_counter.values())
        top_words = sorted(word for word, freq in cluster_word_counter.items() if freq == max_freq)

        base_name = " & ".join(top_words)
        final_name = base_name
        suffix = 2
        while final_name in renamed:
            final_name = f"{base_name} ({suffix})"
            suffix += 1

        renamed[final_name] = {
            "label": label,
            "stems": sorted(set(stems)),
            "top_frequency": max_freq,
            "top_words": top_words,
        }
    return renamed


def merge_named_clusters(renamed_clusters, renamed_noise_clusters):
    """Merges noise-derived clusters into the main set, disambiguating
    any name collisions with a numeric suffix."""
    merged = dict(renamed_clusters)
    for cluster_name, cluster_data in renamed_noise_clusters.items():
        final_name = cluster_name
        suffix = 2
        while final_name in merged:
            final_name = f"{cluster_name} ({suffix})"
            suffix += 1
        merged[final_name] = cluster_data
    return merged


def print_named_clusters(renamed_clusters, title="SEMANTIC CLUSTERS"):
    print(f"\n======================\n{title}\n======================\n")
    for cluster_name in sorted(renamed_clusters.keys()):
        print(f"CLUSTER: {cluster_name}")
        print("STEMS:")
        print(", ".join(renamed_clusters[cluster_name]["stems"]))
        print()


def print_raw_clusters(clusters, title="CLUSTERS"):
    print(f"\n======================\n{title}\n======================\n")
    for label, stems in sorted(clusters.items()):
        print(f"CLUSTER {label}")
        print(", ".join(sorted(set(stems))))
        print()


def _split_oversized_clusters_once(clusters, stem_occurrences, embedder, tokenizer, model, device,
                                    pos_map, max_cluster_size, min_clusters, min_cluster_len, max_synsets):
    """One leaf-mode HDBSCAN splitting pass -- see recluster_large_clusters,
    which calls this repeatedly. A single pass isn't guaranteed to bring
    every resulting piece under max_cluster_size (leaf-mode HDBSCAN can
    still settle on one comparatively large leaf alongside several small
    ones, e.g. observed splitting a real 147-stem cluster into pieces of
    31/5/10/7/5 -- the 31 needed a second pass), which is exactly what
    the wrapper's loop is for."""
    large_clusters = {label: stems for label, stems in clusters.items() if len(set(stems)) > max_cluster_size}
    if not large_clusters:
        return dict(clusters), {}

    stem_to_parent_cluster = {stem: label for label, stems in large_clusters.items() for stem in stems}
    large_stems = sorted({stem for stems in large_clusters.values() for stem in stems})

    large_texts = []
    large_stem_names = []
    for stem in large_stems:
        occurrences = stem_occurrences.get(stem)
        if not occurrences:
            continue

        observed_words = []
        contexts = []
        candidate_definitions = []
        for occurrence in occurrences:
            observed_words.append(occurrence["word"])
            contexts.append(occurrence["sentence"])
            results = glossbert_predict(occurrence, tokenizer, model, device, pos_map, max_synsets)
            if results is None:
                continue
            candidate_definitions.extend(c["definition"] for c in results[:3])

        observed_words = list(dict.fromkeys(observed_words))[:5]
        contexts = list(dict.fromkeys(contexts))[:3]
        candidate_definitions = list(dict.fromkeys(candidate_definitions))[:5]

        prefix = f"Stem: {stem}. Observed words: " + ", ".join(observed_words) + ". Contexts: "
        suffix = ". Candidate senses: " + " ".join(candidate_definitions)
        contexts = _fit_contexts_to_budget(embedder.tokenizer, embedder.max_seq_length, [prefix, suffix], contexts)

        large_texts.append(prefix + " ".join(contexts) + suffix)
        large_stem_names.append(stem)

    large_embeddings = embedder.encode(large_texts, convert_to_numpy=True, normalize_embeddings=True)
    large_labels = hdbscan.HDBSCAN(
        min_cluster_size=min_clusters, min_samples=1, cluster_selection_method="leaf",
    ).fit_predict(large_embeddings)

    large_subclusters = {}
    for stem, label in zip(large_stem_names, large_labels):
        if label == -1:
            continue
        large_subclusters.setdefault(label, set()).add(stem)
    large_subclusters = {label: stems for label, stems in large_subclusters.items() if len(stems) >= min_cluster_len}

    parent_to_subclusters = {}
    for subcluster_label, stems in large_subclusters.items():
        parent_labels = {stem_to_parent_cluster[s] for s in stems if s in stem_to_parent_cluster}
        for parent_label in parent_labels:
            parent_to_subclusters.setdefault(parent_label, []).append(subcluster_label)

    updated_clusters = dict(clusters)
    next_cluster_label = (max(updated_clusters.keys()) + 1) if updated_clusters else 0

    for parent_label, subcluster_labels in parent_to_subclusters.items():
        if len(subcluster_labels) <= 1:
            continue
        updated_clusters.pop(parent_label, None)
        for subcluster_label in subcluster_labels:
            updated_clusters[next_cluster_label] = sorted(large_subclusters[subcluster_label])
            next_cluster_label += 1

    return updated_clusters, large_subclusters


def recluster_large_clusters(clusters, stem_occurrences, embedder, tokenizer, model, device,
                              pos_map=POS_MAP, max_cluster_size=10, min_clusters=5,
                              min_cluster_len=3, max_synsets=5, max_passes=5):
    """Splits any cluster larger than max_cluster_size into finer-grained
    subclusters using richer (word + context + candidate-sense) text
    representations. Returns (updated_clusters, all_subclusters).

    Runs the splitting pass (_split_oversized_clusters_once) repeatedly
    rather than just once -- a single leaf-mode HDBSCAN pass is not
    guaranteed to bring every piece under max_cluster_size (verified on
    real data: a 147-stem cluster's first split left a 31-stem piece
    still over a max_cluster_size of 10) -- but repeating the exact same
    call on data it already failed to split just reproduces the same
    result (HDBSCAN isn't randomized here): re-running
    _split_oversized_clusters_once with an unchanged min_cluster_size on
    a cluster it left alone finds "1 leaf, not promoted" again, every
    time, because nothing about the question changed. So each retry pass
    that makes no progress lowers the HDBSCAN min_cluster_size floor by
    1 (never below min_cluster_len) before trying again -- an actually
    looser question, not the same one repeated -- and stops once nothing
    is left oversized, once min_cluster_size can't drop any further, or
    once max_passes is reached. A cluster can still come out of this
    function over max_cluster_size if the data genuinely doesn't
    separate even at the loosest threshold tried; callers that care
    should check the returned clusters' sizes (find_oversized_clusters)
    rather than assume the cap was met."""
    updated_clusters = dict(clusters)
    all_subclusters = {}
    current_min_clusters = min_clusters
    for pass_num in range(max_passes):
        if not find_oversized_clusters(updated_clusters, max_cluster_size):
            break
        if current_min_clusters < min_cluster_len:
            break

        next_clusters, subclusters = _split_oversized_clusters_once(
            updated_clusters, stem_occurrences, embedder, tokenizer, model, device,
            pos_map=pos_map, max_cluster_size=max_cluster_size, min_clusters=current_min_clusters,
            min_cluster_len=min_cluster_len, max_synsets=max_synsets,
        )

        if next_clusters == updated_clusters:
            # No structural change at this threshold -- loosen it before
            # trying again instead of repeating an identical computation.
            current_min_clusters -= 1
            continue

        updated_clusters = next_clusters
        # HDBSCAN's own labels are only unique within one pass -- prefix
        # by pass number so successive passes' subclusters don't collide
        # and silently overwrite each other in the merged report dict.
        all_subclusters.update({f"{pass_num}.{label}": stems for label, stems in subclusters.items()})
    return updated_clusters, all_subclusters


def find_oversized_clusters(clusters, max_cluster_size):
    """Clusters still over max_cluster_size after recluster_large_clusters
    -- can legitimately happen (see that function's docstring) when the
    data just doesn't separate into smaller pieces within max_passes.
    Callers should surface this rather than let it pass silently, same
    principle as sanity_failed/soft_accept elsewhere in this codebase:
    never silently discard the fact that a limit wasn't actually met."""
    return {label: stems for label, stems in clusters.items() if len(set(stems)) > max_cluster_size}


def extract_noise_stems(stem_names, labels):
    return sorted({stem for stem, label in zip(stem_names, labels) if label == -1})


def recluster_noise(noise_stems, stem_occurrences, embedder, tokenizer, model, device,
                     pos_map=POS_MAP, min_clusters=5, min_cluster_len=3, max_synsets=5):
    """Re-clusters stems HDBSCAN initially treated as noise, in case some
    still form a valid (smaller) semantic group."""
    noise_texts = []
    noise_stem_names = []
    for stem in noise_stems:
        for occurrence in stem_occurrences.get(stem, []):
            results = glossbert_predict(occurrence, tokenizer, model, device, pos_map, max_synsets)
            if results is None:
                continue
            prefix = f"Word: {occurrence['word']}. Sentence: "
            suffix = ". Candidate senses: " + " ".join(c["definition"] for c in results[:3])
            [sentence] = _fit_contexts_to_budget(
                embedder.tokenizer, embedder.max_seq_length, [prefix, suffix], [occurrence["sentence"]],
            )
            noise_texts.append(prefix + sentence + suffix)
            noise_stem_names.append(stem)

    if not noise_texts:
        return {}

    noise_embeddings = embedder.encode(noise_texts, normalize_embeddings=True, convert_to_numpy=True)
    noise_labels = hdbscan.HDBSCAN(min_cluster_size=min_clusters).fit_predict(noise_embeddings)

    noise_clusters = {}
    for stem, label in zip(noise_stem_names, noise_labels):
        if label == -1:
            continue
        noise_clusters.setdefault(label, []).append(stem)

    return {label: stems for label, stems in noise_clusters.items() if len(set(stems)) >= min_cluster_len}


# ============================================================
# N-GRAM MINING
# ============================================================

def tokenize_many_for_analysis(nlp, texts, stemmer, use_stopwords=True, use_lemmas=True):
    sentence_cache = {}
    for doc, text in zip(nlp.pipe(texts), texts):
        tokens = []
        for token in doc:
            if not token.is_alpha:
                continue
            if use_stopwords and token.is_stop:
                continue
            tokens.append(token.lemma_.lower() if use_lemmas else stemmer.stem(token.text.lower()))
        sentence_cache[text] = tokens
    return sentence_cache


def build_sentence_token_cache(nlp, accepted_definitions, stemmer, use_lemmas=True):
    sentences = {
        inst.get("sentence", "").strip()
        for data in accepted_definitions.values()
        for inst in data.get("instances", [])
        if inst.get("sentence", "").strip()
    }
    return tokenize_many_for_analysis(nlp, list(sentences), stemmer, use_stopwords=True, use_lemmas=use_lemmas)


def normalize_cluster_stem(stem, stemmer, use_lemmas):
    stem = stem.rstrip("*").lower()
    return stem if use_lemmas else stemmer.stem(stem)


def token_matches_cluster_stem(token, stems, stemmer, use_lemmas):
    for stem in stems:
        normalized = normalize_cluster_stem(stem, stemmer, use_lemmas)
        if token == normalized or token.startswith(normalized) or normalized.startswith(token):
            return True
    return False


def build_global_ngram_statistics(sentence_cache, min_n=2, max_n=3, min_global_count=2, percentile=95):
    """Corpus-wide n-gram frequency table + the count threshold a gram
    must clear to be eligible as a cluster's "representative" n-gram.

    Hapax grams (count == 1) are excluded before the percentile itself is
    computed -- a single stray occurrence shouldn't be able to pull the
    threshold down -- and the percentile defaults to the 95th (up from
    75th), both of which push the threshold meaningfully higher so only
    genuinely recurring phrases qualify."""
    counter = Counter()
    for tokens in sentence_cache.values():
        for n in range(min_n, max_n + 1):
            if len(tokens) < n:
                continue
            for i in range(len(tokens) - n + 1):
                counter[tuple(tokens[i:i + n])] += 1

    frequencies = [c for c in counter.values() if c >= min_global_count]
    threshold = float(np.percentile(frequencies, percentile)) if frequencies else 0.0
    print(f"Unique ngrams: {len(counter)}")
    print(f"{percentile}th percentile threshold (grams with count >= {min_global_count}): {threshold}")
    return counter, threshold


def _is_subsequence(short_gram, long_gram):
    n = len(short_gram)
    return any(long_gram[i:i + n] == short_gram for i in range(len(long_gram) - n + 1))


def _drop_subsumed_grams(ranked_grams):
    """ranked_grams: [(gram_tuple, count), ...] already ordered by
    priority (length desc, ties broken by count desc) -- longer grams are
    considered first and always get first claim, regardless of whether a
    shorter substring happened to occur more often, so e.g. "social media
    platform" survives and "social media" is dropped as redundant even if
    "social media" alone was individually more frequent."""
    kept = []
    for gram, count in ranked_grams:
        if any(len(kept_gram) > len(gram) and _is_subsequence(gram, kept_gram) for kept_gram, _ in kept):
            continue
        kept.append((gram, count))
    return kept


def extract_cluster_ngrams(clusters, accepted_definitions, sentence_cache, global_ngram_counter,
                            percentile_threshold, stemmer, use_lemmas=True,
                            min_n=2, max_n=3, min_local_count=2):
    """Representative n-grams per cluster: must (a) overlap one of the
    cluster's own stems, (b) clear the global percentile_threshold, (c)
    occur at least min_local_count times within the cluster itself, and
    (d) not be fully contained in another, longer kept gram. Capped to at
    most one n-gram per stem in the cluster, so a cluster's n-gram list
    can't dwarf its own stem list."""
    cluster_ngrams = {}
    for cluster_id, stems in clusters.items():
        local_counter = Counter()
        for stem in stems:
            for inst in accepted_definitions.get(stem, {}).get("instances", []):
                sentence = inst.get("sentence", "").strip()
                tokens = sentence_cache.get(sentence) if sentence else None
                if not tokens:
                    continue
                for n in range(min_n, max_n + 1):
                    if len(tokens) < n:
                        continue
                    for i in range(len(tokens) - n + 1):
                        gram = tuple(tokens[i:i + n])
                        if not any(token_matches_cluster_stem(t, stems, stemmer, use_lemmas) for t in gram):
                            continue
                        if global_ngram_counter.get(gram, 0) < percentile_threshold:
                            continue
                        local_counter[gram] += 1

        candidates = [(gram, count) for gram, count in local_counter.items() if count >= min_local_count]

        # Dedupe by containment first (longer grams get first claim,
        # regardless of count -- see _drop_subsumed_grams), then re-rank
        # the survivors by frequency for the final cap.
        by_length = sorted(candidates, key=lambda gram_count: (len(gram_count[0]), gram_count[1]), reverse=True)
        deduped = _drop_subsumed_grams(by_length)
        deduped.sort(key=lambda gram_count: gram_count[1], reverse=True)

        cap = max(len(set(stems)), 1)
        cluster_ngrams[cluster_id] = [" ".join(g) for g, _ in deduped[:cap]]

    return cluster_ngrams


def attach_ngrams(renamed_clusters, cluster_ngrams):
    for cluster_data in renamed_clusters.values():
        cluster_data["ngrams"] = cluster_ngrams.get(cluster_data["label"], [])
    return renamed_clusters


# ============================================================
# INTERACTIVE CLUSTER REVIEW HELPERS
# ============================================================

def print_cluster_for_review(cluster_name, cluster_data):
    print("\n----------------------------------")
    print(f"CLUSTER NAME: {cluster_name}")
    ngrams = cluster_data.get("ngrams", [])
    if ngrams:
        print("\nRepresentative n-grams:\n")
        for i, gram in enumerate(ngrams, 1):
            print(f"{i}. {gram}")
    print("\nSTEMS:")
    for i, stem in enumerate(cluster_data["stems"], 1):
        print(f"{i}. {stem}")


def parse_index_selection(raw_input, items):
    """Parses a comma-separated 1-based index string (e.g. "2,5,1") into
    (kept, removed) from items. The literal "all" (case-insensitive)
    removes every item. Raises ValueError on malformed input -- callers
    should catch it and fall back to keeping everything, exactly like the
    original inline try/except blocks did."""
    raw_input = raw_input.strip()
    if not raw_input:
        return list(items), []
    if raw_input.lower() == "all":
        return [], list(items)
    indices = {int(x.strip()) - 1 for x in raw_input.split(",")}
    removed = [item for i, item in enumerate(items) if i in indices]
    kept = [item for i, item in enumerate(items) if i not in indices]
    return kept, removed


def run_cluster_review(renamed_clusters, decisions):
    """Interactive per-cluster review: accept as-is, rename/remove
    stems/remove n-grams, or drop the whole cluster entirely. Every
    choice is appended to the decisions log via `decisions.record(...)`
    for later audit -- it does not change the interactive flow. Shared by
    phase1.ipynb and run_pipeline.py so there's one implementation, not
    two. Returns (final_clusters, excluded_cluster_ngrams)."""
    print("\n======================")
    print("CLUSTER REVIEW")
    print("======================\n")

    final_clusters = {}
    excluded_cluster_ngrams = set()

    for cluster_name in sorted(renamed_clusters.keys()):

        cluster_data = renamed_clusters[cluster_name]
        stems = cluster_data["stems"]

        print_cluster_for_review(cluster_name, cluster_data)

        ngrams = cluster_data.get("ngrams", [])
        updated_ngrams = ngrams.copy()
        removed_ngrams = []

        if ngrams:

            remove_ngram_input = input(
                "\nType n-gram numbers to remove (comma-separated), "
                "'all' to remove all, or press ENTER to keep all: "
            ).strip()

            try:
                updated_ngrams, removed_ngrams = parse_index_selection(remove_ngram_input, ngrams)
            except ValueError:
                print("Invalid n-gram selection.")

            excluded_cluster_ngrams.update(removed_ngrams)

            decisions.record(
                step="cluster_review", decision_type="ngram_removal",
                prompt="Type n-gram numbers to remove", options=ngrams,
                choice=remove_ngram_input,
                extra={"cluster_name": cluster_name, "removed_ngrams": removed_ngrams},
            )

        print("\nAccept this cluster, modify it, or remove it entirely?")
        accept = input("(y = accept / n = modify / d = drop entirely): ").strip().lower()

        decisions.record(
            step="cluster_review", decision_type="accept_or_modify",
            prompt="Accept this cluster, modify it, or remove it entirely?", options=["y", "n", "d"],
            choice=accept, extra={"cluster_name": cluster_name},
        )

        if accept == "y":

            final_clusters[cluster_name] = {
                "stems": stems,
                "ngrams": updated_ngrams,
                "excluded_ngrams": removed_ngrams,
                "excluded_stems": [],
            }
            continue

        if accept == "d":

            decisions.record(
                step="cluster_review", decision_type="drop_cluster",
                prompt="Cluster dropped entirely", choice="d",
                extra={"cluster_name": cluster_name},
            )

            print(f"\nCluster '{cluster_name}' dropped entirely.")
            continue

        new_name = input("\nNew cluster name (leave empty to keep current): ").strip()

        decisions.record(
            step="cluster_review", decision_type="rename_cluster",
            prompt="New cluster name", choice=new_name or cluster_name,
            extra={"original_name": cluster_name},
        )

        if new_name == "":
            new_name = cluster_name

        print("\nCurrent stems:")
        for i, stem in enumerate(stems, 1):
            print(f"{i}. {stem}")

        remove_input = input(
            "\nType stem numbers to remove "
            "\n(comma-separated and in any order as in '5,2,3,9...'), "
            "\ntype 'all' to remove all, "
            "\nor press ENTER to keep all: "
        ).strip()

        updated_stems, removed_stems = stems.copy(), []

        try:
            updated_stems, removed_stems = parse_index_selection(remove_input, stems)
        except ValueError:
            print("\nInvalid input. Keeping all stems.")

        decisions.record(
            step="cluster_review", decision_type="stem_removal",
            prompt="Type stem numbers to remove", options=stems,
            choice=remove_input,
            extra={"cluster_name": cluster_name, "removed_stems": removed_stems},
        )

        if removed_stems:
            print("\nRemoved stems:")
            for i, stem in enumerate(removed_stems, 1):
                print(f"{i}. {stem}")

        final_clusters[new_name] = {
            "stems": updated_stems,
            "ngrams": cluster_data.get("ngrams", []),
            "excluded_ngrams": removed_ngrams,
            "excluded_stems": removed_stems,
        }

    print("\n======================")
    print("FINAL CLUSTERS")
    print("======================\n")

    for cluster_name in sorted(final_clusters.keys()):

        print(f"CLUSTER: {cluster_name}")
        print("STEMS:")
        print(", ".join(final_clusters[cluster_name]["stems"]))

        ngrams = final_clusters[cluster_name].get("ngrams", [])
        if ngrams:
            print("\nN-GRAMS:")
            print(", ".join(ngrams[:100]))

        excluded_ngrams = final_clusters[cluster_name].get("excluded_ngrams", [])
        if excluded_ngrams:
            print("\nEXCLUDED N-GRAMS:")
            print(", ".join(excluded_ngrams))

        excluded = final_clusters[cluster_name]["excluded_stems"]
        if excluded:
            print("EXCLUDED STEMS:")
            print(", ".join(excluded))

        print()

    return final_clusters, excluded_cluster_ngrams


# ============================================================
# EXPORT
# ============================================================

def build_phase1_state(final_clusters, excluded_cluster_ngrams):
    return {
        "excludedNgrams": sorted(excluded_cluster_ngrams),
        "clusterDefs": [
            {
                "name": cluster_name,
                "stems": final_clusters[cluster_name]["stems"],
                "ngrams": final_clusters[cluster_name].get("ngrams", []),
                "excluded_stems": final_clusters[cluster_name].get("excluded_stems", []),
                "excluded_ngrams": final_clusters[cluster_name].get("excluded_ngrams", []),
            }
            for cluster_name in sorted(final_clusters.keys())
        ],
    }


def save_phase1_state(state, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=4, ensure_ascii=False)


def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _html_list(label, items, css=""):
    if not items:
        return ""
    joined = ", ".join(f"<code>{item}</code>" for item in items)
    style = f' style="{css}"' if css else ""
    return f"<div{style}><strong>{label}:</strong> {joined}</div>"


def build_cluster_html(final_clusters, corpus_name, tableau20=TABLEAU20):
    rows = []
    for i, cluster_name in enumerate(sorted(final_clusters.keys())):
        cluster_data = final_clusters[cluster_name]
        r, g, b = hex_to_rgb(tableau20[i % len(tableau20)])

        content = (
            _html_list("Stems", cluster_data.get("stems", []))
            + _html_list("N-grams", cluster_data.get("ngrams", []), css="margin-top:8px;")
            + _html_list("Excluded stems", cluster_data.get("excluded_stems", []),
                         css="margin-top:6px;font-size:0.80em;color:#999;")
            + _html_list("Excluded n-grams", cluster_data.get("excluded_ngrams", []),
                         css="margin-top:4px;font-size:0.80em;color:#999;")
        )

        rows.append(
            "    <tr>\n"
            '      <td style="padding:5px 12px 5px 0;">&nbsp;</td>\n'
            f'      <td style="padding:5px 12px 5px 0;color:rgb({r},{g},{b});'
            f'font-weight:bold;vertical-align:top;">{cluster_name}</td>\n'
            f'      <td style="padding:5px 0;font-size:0.88em;color:#555;">{content}</td>\n'
            "    </tr>"
        )

    total_stems = sum(len(c.get("stems", [])) for c in final_clusters.values())
    total_ngrams = sum(len(c.get("ngrams", [])) for c in final_clusters.values())

    return f"""
<h3>Semantic Clusters &mdash; Phase 1 Results</h3>

<p style="font-style:italic;color:#666;font-size:0.9em;">
  {corpus_name} &mdash;
  {len(final_clusters)} clusters &middot;
  {total_stems} stems &middot;
  {total_ngrams} n-grams &middot;
  Tableau20 palette
</p>

<table style="border-collapse:collapse;font-family:serif;font-size:14px;">
  <thead>
    <tr>
      <th style="padding:5px 12px 5px 0;">&nbsp;</th>
      <th style="padding:5px 12px 5px 0;text-align:left;">Cluster</th>
      <th style="padding:5px 0;text-align:left;">Contents</th>
    </tr>
  </thead>
  <tbody>
{chr(10).join(rows)}
  </tbody>
</table>
"""


def save_html(html, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
