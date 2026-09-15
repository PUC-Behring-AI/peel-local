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
from a spaCy-lemmatized, alpha-only, NON-STOPWORD token stream
(build_sentence_token_cache's use_stopwords=True, use_lemmas=True), so an
n-gram matches a contiguous span of that SAME filtered stream, not a
literal contiguous span of the raw document -- a real n-gram commonly has
a stopword or punctuation mark sitting between its content words in the
actual text (e.g. "expertise opacity trust" was mined from "Expertise,
opacity, and trust", with a comma and "and" in between). Matching against
the raw token stream instead of the filtered one -- which this module did
until this fix -- silently missed such occurrences entirely, sometimes
undercounting a real n-gram's frequency all the way to zero even in the
very text it was mined from (verified on real Boisseau data: "artificial
system kind", "expert competence", and "expertise opacity trust" all
showed 0 source occurrences despite being genuine Phase 1 n-grams)."""


def token_forms(doc, stemmer):
    """Returns (stems, lemmas): one PorterStemmer-stemmed string and one
    spaCy-lemma string per alpha, non-stopword token, lowercased --
    empty string at both positions for a non-alpha OR stopword token
    (same treatment for both, mirroring how Phase 1 filters its own
    ngram-mining stream), so an empty/blank term can never accidentally
    match one, and n-gram matching (term_match_spans/term_positions) can
    skip straight past either kind of gap. Deliberately still one entry
    per REAL document token (not a compacted list) -- callers that need
    to highlight a match (build_reader_html) need real token indices to
    wrap a <mark> around the actual word, not an index into some
    filtered-out subset's own numbering."""
    stems, lemmas = [], []
    for token in doc:
        if token.is_alpha and not token.is_stop:
            stems.append(stemmer.stem(token.text.lower()))
            lemmas.append(token.lemma_.lower())
        else:
            stems.append("")
            lemmas.append("")
    return stems, lemmas


def _compact_lemma_positions(lemmas):
    """(real_index, lemma) for every non-empty (content) lemma slot --
    empty slots are non-alpha or stopword tokens (see token_forms), both
    skipped here so an n-gram search below can find a contiguous run of
    CONTENT words regardless of what filler sits between them in the real
    document, the same way Phase 1 originally mined it from an
    equivalently filtered stream."""
    return [(i, lemma) for i, lemma in enumerate(lemmas) if lemma]


def term_match_spans(stems, lemmas, term):
    """Like term_positions, but returns each match's full tuple of REAL
    token indices -- length 1 for a stem, length len(words) for an
    n-gram -- instead of collapsing every match down to one anchor
    position. Needed for highlighting the actual matched words: for an
    n-gram, those words are not guaranteed to be contiguous in the real
    token stream (a stopword/punctuation gap may sit between them), so a
    naive range(start, start+n) would highlight the wrong tokens -- or,
    if the real gap is wider than the n-gram itself, miss the later
    word(s) entirely."""
    normalized = term.rstrip("*").lower()
    if not normalized:
        return []
    if " " in normalized:
        words = normalized.split()
        n = len(words)
        compact = _compact_lemma_positions(lemmas)
        compact_lemmas = [lemma for _, lemma in compact]
        return [
            tuple(compact[i + k][0] for k in range(n))
            for i in range(len(compact_lemmas) - n + 1)
            if compact_lemmas[i:i + n] == words
        ]
    return [(i,) for i, s in enumerate(stems) if s == normalized]


def term_positions(stems, lemmas, term):
    """term: a raw cluster stem (possibly with a trailing '*') or n-gram
    string, exactly as stored in clusterDefs. Returns one anchor position
    per match -- the first matched word's real token index -- which is
    all counting (len(...)), proximity, and binning callers need,
    matching this function's pre-existing contract exactly. See
    term_match_spans for a match's full real-index span, needed only for
    highlighting."""
    return [span[0] for span in term_match_spans(stems, lemmas, term)]


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
