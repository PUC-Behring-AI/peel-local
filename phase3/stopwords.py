"""Comprehensive stopword construction for Phase 3's frequency-based
tools (word clouds, term-frequency tables). Phase 1's own excludedNgrams
plus spaCy's built-in English stopword list is often too narrow for a
useful word cloud -- numerals and citation-abbreviation artifacts
otherwise dominate.

Adapted from a researcher-supplied specification for a separate,
Voyant-based PEEL system this repo doesn't use: that spec required a
researcher to confirm candidate author-name exclusions by hand, since a
heuristic name extractor has a real false-positive rate. Here, candidate
names are included automatically once they clear a minimum-mention
threshold, and always disclosed in the report's provenance section
rather than silently applied -- an automatic proxy for "probably a cited
author," not a researcher-confirmed judgment call.
"""

import re
from collections import Counter

NUMERAL_RE = re.compile(r"^\d+[a-z]?$")  # "1991", "2024a", "41"

CITATION_ABBREVIATIONS = {
    "et", "al", "ibid", "eg", "ie", "etc", "cf", "op", "cit", "vol", "no", "pp",
}


def numeral_stopwords(text):
    return {w.lower() for w in re.findall(r"\b\w+\b", text) if NUMERAL_RE.match(w)}


def candidate_author_surnames(doc):
    """PERSON entities spaCy finds, ranked by mention count."""
    counts = Counter()
    for ent in doc.ents:
        if ent.label_ != "PERSON":
            continue
        for token in ent:
            if token.is_alpha and token.text[:1].isupper():
                counts[token.text] += 1
    return counts.most_common()


def build_stopwords(nlp_stopwords, doc, min_author_mentions=3):
    """Combines spaCy's built-in English stopword list with the
    automatic numeral/citation-abbreviation rules and any PERSON-entity
    name mentioned at least min_author_mentions times. Returns
    (stopword_set, auto_included_authors, all_candidates) -- the caller
    discloses the auto-included subset in the report rather than
    treating this as a silent default."""
    text = doc.text
    stopwords = {w.lower() for w in nlp_stopwords}
    stopwords |= numeral_stopwords(text)
    stopwords |= CITATION_ABBREVIATIONS

    candidates = candidate_author_surnames(doc)
    auto_authors = [name for name, count in candidates if count >= min_author_mentions]
    stopwords |= {name.lower() for name in auto_authors}

    return stopwords, auto_authors, candidates
