"""Shared term-matching utilities for Phase 3: finding a cluster
stem/n-gram's token positions in a document, and a proximity check
between two position lists. Used by both collocations.py (the empirical
collocation scan) and distant_reading.py (Contexts/Collocates/Reader),
so there's one implementation of "does this term occur here" instead of
several slightly-different ones.

Stems and n-grams are matched differently, mirroring how phase1/pipeline.py
itself builds them: a cluster's `stems` are PorterStemmer output (see
phase1/pipeline.py::extract_top_stems), so a stem matches a token by exact
equality against that token's own stem. A cluster's `ngrams` are mined
from spaCy-lemmatized token sequences (build_sentence_token_cache's
use_lemmas=True), so an n-gram matches a contiguous span by exact
equality against each token's lemma, not its stem.
"""


def token_forms(doc, stemmer):
    """Returns (stems, lemmas): one PorterStemmer-stemmed string and one
    spaCy-lemma string per alpha token, lowercased -- empty string at
    both positions for a non-alpha token, so an empty/blank term can
    never accidentally match one."""
    stems, lemmas = [], []
    for token in doc:
        if token.is_alpha:
            stems.append(stemmer.stem(token.text.lower()))
            lemmas.append(token.lemma_.lower())
        else:
            stems.append("")
            lemmas.append("")
    return stems, lemmas


def term_positions(stems, lemmas, term):
    """term: a raw cluster stem (possibly with a trailing '*') or n-gram
    string, exactly as stored in clusterDefs. Single-word stems match by
    exact equality against a token's own stem; multi-word n-grams match a
    contiguous span by exact equality against each word's lemma."""
    normalized = term.rstrip("*").lower()
    if not normalized:
        return []
    if " " in normalized:
        words = normalized.split()
        n = len(words)
        return [i for i in range(len(lemmas) - n + 1) if lemmas[i:i + n] == words]
    return [i for i, s in enumerate(stems) if s == normalized]


def count_positions_near(pos_a, pos_b, proximity_n):
    """How many entries in pos_a (sorted ascending) have at least one
    entry in pos_b (sorted ascending) within proximity_n token positions
    -- a two-pointer sweep, O(len(pos_a)+len(pos_b)), since a frequent
    term can have hundreds of positions in a real corpus."""
    j = 0
    hits = 0
    for pa in pos_a:
        while j < len(pos_b) and pos_b[j] < pa - proximity_n:
            j += 1
        k = j
        found = False
        while k < len(pos_b) and pos_b[k] <= pa + proximity_n:
            if pos_b[k] != pa:
                found = True
                break
            k += 1
        if found:
            hits += 1
    return hits


def positions_near(pos_a, target_positions, proximity_n):
    """Filters pos_a down to just the entries within proximity_n of any
    entry in target_positions -- used by Contexts to restrict a KWIC
    listing to genuine collocation occurrences, not every occurrence of
    the first term alone."""
    return [
        pa for pa in pos_a
        if any(abs(pa - pb) <= proximity_n and pa != pb for pb in target_positions)
    ]
