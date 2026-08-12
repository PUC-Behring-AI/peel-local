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
    match = re.search(re.escape(word), sentence, re.IGNORECASE)
    marked_sentence = (
        f'{sentence[:match.start()]} "{match.group()}" {sentence[match.end():]}'
        if match else sentence
    )

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


# ============================================================
# STEM EXTRACTION & MAPPING
# ============================================================

def extract_top_stems(doc, stemmer, top_percentile, max_stems) -> dict:
    tokens = [
        stemmer.stem(token.text.lower())
        for token in doc
        if not token.is_stop and not token.is_punct and not token.is_space and token.is_alpha
    ]

    sorted_freq = sorted(Counter(tokens).items(), key=lambda x: x[1], reverse=True)
    final_n = min(math.ceil(len(sorted_freq) * top_percentile), max_stems)
    return dict(sorted_freq[:final_n])


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


def run_glossbert_analysis(top_stems, stem_occurrences, tokenizer, model, device,
                            pos_map=POS_MAP, max_sentences_per_stem=5, max_synsets=5):
    accepted_definitions = {}
    flagged_words = []

    for stem, count in tqdm(top_stems.items(), total=len(top_stems), desc="GlossBERT"):
        occurrences = stem_occurrences.get(stem, [])[:max_sentences_per_stem]
        if not occurrences:
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
    "accept_all", "exit", or "invalid".

    Candidate slots are 1..len(top_candidates); the manual-entry slot is
    whatever number comes right after the last candidate -- both derived
    from the actual candidate count (driven by max_synsets) rather than
    a fixed "3 candidates + slot 4" assumption."""
    choice = choice.strip()
    if choice.lower() == "accept all":
        return "accept_all", None
    if choice.lower() == "exit":
        return "exit", None
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


def run_flagged_term_review(flagged_words, accepted_definitions, decisions):
    """Interactive review of GlossBERT's flagged sense mismatches: accept
    all predictions, review one by one, or select specific terms. Mutates
    accepted_definitions in place. Every choice is appended to the
    decisions log via `decisions.record(...)` for later audit -- it does
    not change the interactive flow. Shared by phase1.ipynb and
    run_pipeline.py so there's one implementation, not two."""
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

        stem_texts.append(
            f"Stem: {stem}. Observed words: " + ", ".join(observed_words)
            + ". Contexts: " + " ".join(contexts)
            + ". Candidate senses: " + " ".join(definitions)
        )
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

        renamed[" & ".join(top_words)] = {
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

        large_texts.append(
            f"Stem: {stem}. Observed words: " + ", ".join(observed_words)
            + ". Contexts: " + " ".join(contexts)
            + ". Candidate senses: " + " ".join(candidate_definitions)
        )
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
            noise_texts.append(
                f"Word: {occurrence['word']}. Sentence: {occurrence['sentence']}. "
                "Candidate senses: " + " ".join(c["definition"] for c in results[:3])
            )
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
