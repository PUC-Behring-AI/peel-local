"""Corpus-size ceiling, and the one place spaCy is loaded.

spaCy refuses any text over `nlp.max_length` (1,000,000 characters by
default) with error E088, whose message talks about an attribute a
researcher has no reason to know exists. Worse, the failure lands *after*
the corpus has been read and, for Phase 2, after Phase 1's whole GPU pass.
Every `spacy.load` in this codebase therefore goes through `load_spacy`
below, which raises the ceiling to one named value and keeps the
translation from "E088" to "your corpus is too long, here is the actual
number" in a single place.

The ceiling is not free. spaCy's own default exists because the parser and
NER allocate roughly a gigabyte per 100,000 characters, and
`phase2/pipeline.py::segment_corpus` then holds one `Doc` per sentence for
the whole corpus on top of that. `describe_corpus_size` exists to say so
before a researcher discovers it as a swap-thrashing machine: it returns a
note (or None when there is nothing worth saying), following the same
"return a note the caller prints" convention as
`phase1/pipeline.py::describe_stem_cap` and `describe_wordnet_excluded_stems`.

Raising the ceiling changes nothing for a corpus below spaCy's own default
-- which includes every corpus committed under `data/` (the worked
example's source is 83,164 characters). It only changes what happens to a
corpus that previously crashed.
"""

# spaCy's own default, kept as a named value because it is the threshold the
# memory warning below is keyed to, not an arbitrary number.
SPACY_DEFAULT_MAX_LENGTH = 1_000_000

# What this pipeline actually allows. Chosen as a doubling of spaCy's default
# rather than "as high as possible": beyond roughly this, segment_corpus's
# per-sentence Doc cache is the binding constraint, not spaCy's guard, so a
# higher number here would just move the failure somewhere less legible.
SPACY_MAX_LENGTH = 2_000_000


def load_spacy(model_name: str):
    """Loads a spaCy pipeline with this project's corpus-size ceiling applied.

    Every caller in the codebase uses this instead of `spacy.load` directly,
    so the ceiling cannot drift between the CLI, the webapp, and the
    notebooks. Raises a `SystemExit` with an actionable message if the model
    is not installed, rather than spaCy's own E050 traceback.
    """
    import spacy

    try:
        nlp = spacy.load(model_name)
    except OSError as e:
        raise SystemExit(
            f"spaCy model {model_name!r} is not installed. Install it with:\n"
            f"    python -m spacy download {model_name}\n"
            f"(original error: {e})"
        ) from e

    nlp.max_length = SPACY_MAX_LENGTH
    return nlp


def describe_corpus_size(text: str, max_length: int = SPACY_MAX_LENGTH) -> str | None:
    """One-line note about a corpus large enough for size to matter, or None.

    Two thresholds, because they have different consequences:

    - Over `max_length`, nothing will work: spaCy refuses the text outright.
      The note names the ceiling and the actual length so the researcher can
      split the corpus rather than guess at what "too long" means.
    - Over spaCy's own default but under the ceiling, it will work but may be
      slow or memory-hungry, chiefly because `segment_corpus` keeps one
      parsed `Doc` per sentence. Worth saying before the machine starts
      swapping, not after.
    """
    n = len(text)
    if n > max_length:
        return (
            f"ERROR: this corpus is {n:,} characters, over the {max_length:,}-character "
            "ceiling this pipeline sets on spaCy (common/limits.py::SPACY_MAX_LENGTH). "
            "Split it into smaller corpora and analyse them separately, or raise the "
            "ceiling there if you have the memory for it."
        )
    if n > SPACY_DEFAULT_MAX_LENGTH:
        return (
            f"NOTE: this corpus is {n:,} characters, above spaCy's own default "
            f"{SPACY_DEFAULT_MAX_LENGTH:,}-character limit (raised to {max_length:,} here). "
            "Expect high memory use: spaCy's parser/NER allocate roughly 1GB per 100,000 "
            "characters, and Phase 2 keeps one parsed sentence object per sentence on top "
            "of that."
        )
    return None
