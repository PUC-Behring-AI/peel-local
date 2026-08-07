"""PEEL Phase 2 pipeline: selects the most informative sentences per
Phase 1 cluster, using lexical density plus stem/n-gram matching.

Phase 2 has no interactive review steps (unlike phase1), so there's no
decision-log wiring here -- the notebook is a thin orchestration layer
over these functions.
"""

import json

import numpy as np
from spacy.matcher import PhraseMatcher


def load_phase1_state(path):
    with open(path, encoding="utf8") as f:
        return json.load(f)


def segment_corpus(nlp, text):
    doc = nlp(text)

    sentence_dict = {}
    sentence_docs = {}
    for i, sent in enumerate(doc.sents):
        sentence = sent.text.strip()
        if not sentence:
            continue
        sentence_dict[i] = sentence
        sentence_docs[i] = nlp(sentence)

    return sentence_dict, sentence_docs


def lexical_density(doc):
    tokens = [t for t in doc if not t.is_space]
    if not tokens:
        return 0.0
    content = [t for t in tokens if not t.is_stop and not t.is_punct]
    return len(content) / len(tokens)


def compute_lexical_densities(sentence_docs, percentile):
    densities = {idx: lexical_density(doc) for idx, doc in sentence_docs.items()}
    threshold = np.percentile(list(densities.values()), percentile)
    return densities, threshold


def stem_matches(doc, cluster_stems, stemmer):
    sentence_stems = {stemmer.stem(token.text.lower()) for token in doc if token.is_alpha}
    return {stem for stem in cluster_stems if stem.rstrip("*").lower() in sentence_stems}


def build_phrase_matcher(nlp, ngrams):
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    if ngrams:
        matcher.add("NGRAMS", [nlp.make_doc(g) for g in ngrams])
    return matcher


def highlight_sentence(doc, matched_stems, matched_ngrams, nlp, stemmer):
    """Highlights matched stems and n-grams using spaCy token indices.
    N-grams take priority over stems."""
    matcher = build_phrase_matcher(nlp, matched_ngrams)
    matches = matcher(doc)

    highlight = [False] * len(doc)
    for _, start, end in matches:
        for i in range(start, end):
            highlight[i] = True

    normalized_cluster = {stem.rstrip("*").lower() for stem in matched_stems}
    for token in doc:
        if highlight[token.i] or not token.is_alpha:
            continue
        if stemmer.stem(token.text.lower()) in normalized_cluster:
            highlight[token.i] = True

    pieces = []
    for token, flag in zip(doc, highlight):
        text = f"****{token.text}****" if flag else token.text
        pieces.append(text + token.whitespace_)
    return "".join(pieces)


def score_sentence(density, matched_stems, matched_ngrams):
    return density * (1 + len(matched_stems) + len(matched_ngrams))


def select_representative_sentences(cluster, sentence_docs, densities, density_threshold,
                                     nlp, stemmer, top_n):
    stems = cluster.get("stems", [])
    ngrams = cluster.get("ngrams", [])
    matcher = build_phrase_matcher(nlp, ngrams)

    candidates = []
    for idx, sent_doc in sentence_docs.items():
        density = densities[idx]
        if density < density_threshold:
            continue

        matched_stems = stem_matches(sent_doc, stems, stemmer)
        matched_ngrams = {sent_doc[start:end].text for _, start, end in matcher(sent_doc)}

        if not matched_stems and not matched_ngrams:
            continue

        candidates.append({
            "index": idx,
            "score": round(score_sentence(density, matched_stems, matched_ngrams), 4),
            "lexical_density": round(density, 4),
            "matched_stems": sorted(matched_stems),
            "matched_ngrams": sorted(matched_ngrams),
            "sentence": highlight_sentence(sent_doc, matched_stems, matched_ngrams, nlp, stemmer),
        })

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:top_n]


def enrich_with_informative_sentences(phase1_state, text, nlp, stemmer, top_n, density_percentile):
    print("Segmenting corpus...")
    _, sentence_docs = segment_corpus(nlp, text)
    print(f"{len(sentence_docs)} sentences found.")

    densities, density_threshold = compute_lexical_densities(sentence_docs, density_percentile)
    print(
        f"Lexical density threshold ({density_percentile}th percentile): "
        f"{density_threshold:.3f}"
    )

    print("Selecting representative sentences...")
    for cluster in phase1_state["clusterDefs"]:
        cluster["most_insightful_sentences"] = select_representative_sentences(
            cluster, sentence_docs, densities, density_threshold, nlp, stemmer, top_n,
        )

    return phase1_state


def save_informative_sentences(state, path):
    with open(path, "w", encoding="utf8") as f:
        json.dump(state, f, indent=4, ensure_ascii=False)
