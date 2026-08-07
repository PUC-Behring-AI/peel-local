"""PEEL Phase 0: corpus cleaning.

Removes page numbers, running headers/footers, footnote-call digits,
broken hyphenation, front matter (mastheads, copyright notices, abstracts),
and end sections (notes/references) from a raw .txt corpus.

Usage:
    python clean_corpus.py raw_corpus.txt
    python clean_corpus.py raw_corpus.txt -o cleaned_corpus.txt

Also importable (used by phase0.ipynb and run_pipeline.py):
    from phase0.clean_corpus import clean_text
"""

import re
import argparse
from collections import Counter

import nltk
from nltk.corpus import wordnet as wn

# Download WordNet if necessary
try:
    wn.ensure_loaded()
except LookupError:
    nltk.download("wordnet")
    nltk.download("omw-1.4")

WORDNET = set(w.lower() for w in wn.all_lemma_names())


# ---------------------------------------------------------------------
# DETECTION FUNCTIONS
# ---------------------------------------------------------------------

def remove_isolated_page_numbers(lines):
    return [
        line for line in lines
        if not re.fullmatch(r"\s*\d{1,4}\s*", line)
    ]


def remove_recurring_short_lines(lines, min_repeats=3, max_len=80):
    def normalize(s):
        return re.sub(r"\d+", "N", s.strip())

    normalized = [normalize(x) for x in lines]

    counts = Counter(
        n for n in normalized
        if n and len(n) <= max_len
    )

    recurring = {
        n for n, c in counts.items()
        if c >= min_repeats
    }

    cleaned = []

    for line, norm in zip(lines, normalized):
        if norm in recurring:
            continue
        cleaned.append(line)

    return cleaned


def remove_footnote_calls(text):
    """
    knowledge1 -> knowledge
    topic.2 -> topic.
    network10 -> network
    """
    pattern = re.compile(r'(?<!\d)([A-Za-z)\]\'’.,;:!?])(\d{1,2})(?!\d)')
    return pattern.sub(r"\1", text)


def fix_broken_hyphenation(text):
    """
    epi-
    stemic
      ->
    epistemic
    """
    return re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)


def split_juxtaposed_words(text):
    """
    Uses WordNet to split
    domainindependent -> domain independent
    whenever both parts are valid words.
    """

    tokens = re.findall(r"\b[A-Za-z]{8,}\b", text)

    replacements = {}

    for token in tokens:

        lower = token.lower()

        if lower in WORDNET:
            continue

        for i in range(3, len(lower)-3):

            left = lower[:i]
            right = lower[i:]

            if left in WORDNET and right in WORDNET:
                replacements[token] = token[:i] + " " + token[i:]
                break

    for old, new in replacements.items():
        text = re.sub(rf"\b{re.escape(old)}\b", new, text)

    return text


# ---------------------------------------------------------------------
# FRONT MATTER
# ---------------------------------------------------------------------

FRONT_MATTER_PATTERNS = [
    ('journal_masthead', re.compile(r'^(ORIGINAL RESEARCH|[A-Z][a-z]+\s+\(\d{4}\)|https://doi\.org/)', re.I)),
    ('article_history', re.compile(r'^Received:.*Accepted:', re.I)),
    ('copyright_notice', re.compile(r'^©|under exclusive licence|Springer Nature', re.I)),
    ('acm_reference_format', re.compile(r'^ACM Reference Format', re.I)),
    ('ccs_concepts', re.compile(r'^CCS Concepts', re.I)),
    ('keywords_heading', re.compile(r'^Keywords\b', re.I)),
    ('abstract_heading', re.compile(r'^Abstract\b', re.I)),
    ('email', re.compile(r'[\w.+-]+@[\w-]+\.[\w.-]+')),
    ('affiliation', re.compile(r'Universit|CNRS|Department of', re.I)),
]

def scan_front_matter(lines, body_heading_pattern=r'^(?:\d+(?:\.\d+)?\.?\s*)?Introduction\b'):
    """NOTE (fixed, live evidence, Boisseau 2026 verification, 2026-07-15):
    the first draft of this scanner searched only the span before the
    first body heading, on the assumption front matter is confined there.
    Run for real, this corpus's own extraction injected its ARTICLE
    HISTORY/copyright/author block *after* the Introduction heading and
    mid-sentence -- confirming, on live data this file had not previously
    tested against, the "not reliably positioned" warning above. Scans the
    whole document by content match now, not a bounded span."""
    body_start = None
    for i, line in enumerate(lines):
        if re.match(body_heading_pattern, line.strip()):
            body_start = i
            break

    hits = []
    for i, line in enumerate(lines):
        for tag, pat in FRONT_MATTER_PATTERNS:
            if pat.search(line):
                hits.append((i + 1, tag, line.strip()))
    return hits, body_start

def remove_front_matter(text):
    lines = text.splitlines()

    _, body_start = scan_front_matter(lines)

    if body_start is None:
        body_start = len(lines)

    remove = set()

    for i, line in enumerate(lines[:body_start]):
        for _, pattern in FRONT_MATTER_PATTERNS:
            if pattern.search(line):
                remove.add(i)
                break

    return "\n".join(
        line
        for i, line in enumerate(lines)
        if i not in remove
    )


# ---------------------------------------------------------------------
# REMOVE NOTES / REFERENCES
# ---------------------------------------------------------------------

END_SECTION_PATTERNS = [
    r"^\s*notes\s*$",
    r"^\s*footnotes?\s*$",
    r"^\s*references\s*$",
    r"^\s*bibliography\s*$",
]


def remove_end_sections(text):

    lines = text.splitlines()

    start = None

    for i, line in enumerate(lines):

        if any(
            re.match(p, line, flags=re.I)
            for p in END_SECTION_PATTERNS
        ):
            start = i
            break

    if start is None:
        return text

    return "\n".join(lines[:start])


# ---------------------------------------------------------------------
# WHITESPACE
# ---------------------------------------------------------------------

def normalize_whitespace(text):

    text = re.sub(r"[ \t]+", " ", text)

    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip() + "\n"


# ---------------------------------------------------------------------
# PIPELINE
# ---------------------------------------------------------------------

def clean_text(text):

    # broken words first
    text = fix_broken_hyphenation(text)

    # remove footnote digits
    text = remove_footnote_calls(text)

    # split merged words
    text = split_juxtaposed_words(text)

    # remove front matter
    text = remove_front_matter(text)

    # remove notes/references
    text = remove_end_sections(text)

    # page numbers / headers
    lines = text.splitlines()

    lines = remove_isolated_page_numbers(lines)

    lines = remove_recurring_short_lines(lines)

    text = "\n".join(lines)

    text = normalize_whitespace(text)

    return text


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "input",
        help="Input TXT file"
    )

    parser.add_argument(
        "-o",
        "--output",
        help="Output TXT file",
        default="cleaned_corpus.txt"
    )

    args = parser.parse_args()

    with open(args.input, encoding="utf8") as f:
        text = f.read()

    cleaned = clean_text(text)

    with open(args.output, "w", encoding="utf8") as f:
        f.write(cleaned)

    print(f"Saved cleaned corpus to: {args.output}")


if __name__ == "__main__":
    main()
