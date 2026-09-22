"""The three different answers this pipeline gives to "does this cluster term
occur here", named and set side by side.

There are three, they disagree, and until this module existed none of them was
named -- each was an unlabelled expression inside the function that used it,
so the disagreement was invisible unless you happened to read all three.

| # | Strategy | Stem match | N-gram match | Used by |
|---|---|---|---|---|
| 1 | `stem_prefix_match` | bidirectional prefix | n/a | Phase 1 n-gram mining (`extract_cluster_ngrams`) |
| 2 | `phase3/terms.py::term_positions` | exact stem equality | contiguous run of the lemmatised, stopword-filtered stream | all of Phase 3 except the trend chart; Phase 2 sentence selection |
| 3 | `ngram_substring_count` | n/a (callers use exact stems) | raw substring of the lowercased text | cluster coverage, and the Phase 3 trend chart |

**Why they differ, and which one is right.** Phase 1 mines a cluster's n-grams
from a spaCy-lemmatised, alpha-only, non-stopword token stream. So a mined
n-gram is a reconstruction, not a quotation: `"expertise opacity trust"` came
from "Expertise, opacity, and trust", with a comma and an "and" between its
words. Strategy 2 knows this and matches against the same filtered stream,
which is correct. Strategy 3 does not, and looks for the literal string --
which usually is not there. Measured on the committed example corpus, 22 of
33 cluster n-grams score zero under strategy 3, and 46% of all n-gram
occurrences in the source are lost.

Strategy 1 is the loosest of the three in the other direction: matching on a
prefix in *either* direction means the stem `art` matches the tokens
`article`, `artificial` and `artist`. It exists because a Porter stem like
`observ` has to match the lemma `observation`, and a one-directional prefix
test would not.

**Nothing here changes which strategy any caller uses.** This module holds
the implementations and the comparison; the call sites are unchanged and
produce identical results, asserted by the golden tests under `tests/`.
Unifying them on strategy 2 is tracked as L1 in `LIMITATIONS.md`, and waits
behind the `v1.0.0` behaviour freeze because it moves numbers the paper
reports. The point of naming them first is that the fix then becomes a
one-line change at two call sites, reviewable against a test that says
exactly what moved.
"""


def normalize_cluster_term(term: str, stemmer=None, use_lemmas: bool = True) -> str:
    """A cluster stem or n-gram reduced to the form the matchers compare
    against: trailing `*` dropped (clusterDefs sometimes carries one as a
    wildcard marker) and lowercased, then Porter-stemmed unless the caller is
    working in lemma space."""
    term = term.rstrip("*").lower()
    return term if use_lemmas else stemmer.stem(term)


def stem_prefix_match(token: str, stems, stemmer=None, use_lemmas: bool = True) -> bool:
    """Strategy 1: does `token` match any of `stems` by prefix, either way round?

    True when the token equals a normalized stem, starts with it, or is itself
    a prefix of it. The second direction is what lets a Porter stem (`observ`)
    match a longer lemma (`observation`); the third is what makes this the
    loosest of the three strategies -- `art` matches `article` and
    `artificial`, and `ai` matches `aim`, `air` and `aid`.

    Used only by Phase 1's n-gram mining, where it decides which candidate
    n-grams count as overlapping a cluster's own vocabulary. Because it is
    applied at selection time rather than at counting time, its looseness
    widens the candidate pool rather than inflating a reported figure.
    """
    for stem in stems:
        normalized = normalize_cluster_term(stem, stemmer, use_lemmas)
        if token == normalized or token.startswith(normalized) or normalized.startswith(token):
            return True
    return False


def ngram_substring_count(text_lower: str, ngram: str) -> int:
    """Strategy 3: how many times does `ngram` occur as a literal substring?

    `text_lower` must already be lowercased -- the caller lowercases once and
    reuses it across every term, which is why that is not done here.

    This undercounts, and knowing by how much is the point of having it named:
    a mined n-gram is a lemma reconstruction, so the literal string usually
    does not appear in the text at all (see this module's docstring for the
    measured figures). Kept as the implementation of what cluster coverage and
    the Phase 3 trend chart currently do, so that replacing it with strategy 2
    is a change at one place rather than an edit inside two unrelated
    functions.
    """
    return text_lower.count(normalize_cluster_term(ngram))
