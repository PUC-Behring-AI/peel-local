"""PEEL Phase 2 pipeline: selects the most informative sentences per
Phase 1 cluster, using lexical density plus stem/n-gram matching.

Phase 2 has no interactive review steps (unlike phase1), so there's no
decision-log wiring here -- the notebook is a thin orchestration layer
over these functions.
"""

import json

import numpy as np


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


def _lemma_tokens(doc):
    """Filtered, lemmatized token stream for a sentence -- alpha-only,
    stopwords removed, lemma lowercased -- paired with each entry's real
    doc token index. Mirrors phase1_pipeline.tokenize_many_for_analysis's
    exact filtering (the use_stopwords=True, use_lemmas=True call
    build_sentence_token_cache uses), so an n-gram mined there from a
    filtered/lemmatized stream can be matched back here the same way."""
    lemmas, indices = [], []
    for token in doc:
        if not token.is_alpha or token.is_stop:
            continue
        lemmas.append(token.lemma_.lower())
        indices.append(token.i)
    return lemmas, indices


def ngram_matches(doc, ngrams):
    """Matches cluster n-grams (lemma-joined strings, e.g. "trust human",
    built by extract_cluster_ngrams from lemmatized tokens) against a
    sentence by lemma, not literal text. A literal-text search (spaCy's
    PhraseMatcher over raw surface text, the previous approach) almost
    never finds these: the n-gram string is a lemma reconstruction, not a
    quote, so it frequently doesn't exist verbatim anywhere in the corpus
    even when the underlying phrase (in some inflected form) does --
    verified on real Boisseau data, 17 of 25 cluster n-grams could never
    match ANY sentence under literal matching because they don't occur as
    that exact literal substring anywhere in the raw text. Returns
    (matched_ngram_strings, set_of_matched_doc_token_indices) -- the
    indices are real, verbatim doc tokens (for highlighting the actual
    words that occurred), even though the returned strings are the
    canonical lemma form (for scoring/display, matching how cluster
    n-grams are already shown elsewhere)."""
    if not ngrams:
        return set(), set()

    lemmas, indices = _lemma_tokens(doc)
    matched = set()
    matched_indices = set()
    for gram in ngrams:
        gram_words = gram.lower().split()
        n = len(gram_words)
        if n == 0 or n > len(lemmas):
            continue
        for i in range(len(lemmas) - n + 1):
            if lemmas[i:i + n] == gram_words:
                matched.add(gram)
                matched_indices.update(indices[i:i + n])
    return matched, matched_indices


def highlight_sentence(doc, matched_stems, matched_ngram_indices, stemmer):
    """Highlights matched stems and n-grams using spaCy token indices.
    N-grams take priority over stems."""
    highlight = [False] * len(doc)
    for i in matched_ngram_indices:
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
                                     stemmer, top_n):
    stems = cluster.get("stems", [])
    ngrams = cluster.get("ngrams", [])

    candidates = []
    for idx, sent_doc in sentence_docs.items():
        density = densities[idx]
        if density < density_threshold:
            continue

        matched_stems = stem_matches(sent_doc, stems, stemmer)
        matched_ngrams, matched_ngram_indices = ngram_matches(sent_doc, ngrams)

        if not matched_stems and not matched_ngrams:
            continue

        candidates.append({
            "index": idx,
            "score": round(score_sentence(density, matched_stems, matched_ngrams), 4),
            "lexical_density": round(density, 4),
            "matched_stems": sorted(matched_stems),
            "matched_ngrams": sorted(matched_ngrams),
            "sentence": highlight_sentence(sent_doc, matched_stems, matched_ngram_indices, stemmer),
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
            cluster, sentence_docs, densities, density_threshold, stemmer, top_n,
        )

    return phase1_state


def save_informative_sentences(state, path):
    with open(path, "w", encoding="utf8") as f:
        json.dump(state, f, indent=4, ensure_ascii=False)
