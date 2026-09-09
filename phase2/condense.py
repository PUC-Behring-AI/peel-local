"""Phase 2 condensation: generates an LLM-condensed version of the corpus
at a target word-count rate, then verifies it -- a verbatim-overlap scan,
a deterministic F/T/R/C injection-risk classification of every non-verbatim
sentence, a borderline-classification flag pass, and per-cluster term
coverage against Phase 1's clusters.

Adapted from a researcher-supplied specification for a separate, more
elaborate PEEL system this repo doesn't implement. Two deliberate
adaptations from that spec:

1. Classification is **pure deterministic Python heuristics**, not an LLM
   judgment call -- reproducible, but cruder than a semantic reading.
   Ambiguous cases default to the more conservative label (C over R, T
   over F), same as the source spec's own tie-break rule.
2. The source spec's classification procedure mixes token-level verbatim
   scanning with sentence-level classification rules, which only coheres
   if an LLM is doing the actual span segmentation by reading the text --
   not viable under deterministic heuristics. Here, classification runs
   at the sentence level throughout: `scan_verbatim_overlap` still reports
   the token-level aggregate statistic, but `classify_condensation` makes
   one F/T/R/C decision per condensed sentence.

Interactive control flow (rate/model/trial-count prompts, the escalation
question, borderline-reclassification prompts) lives here as
`run_condensation_setup`/`run_generate_for_rate`/`run_injection_review`,
so phase2.ipynb and run_pipeline.py share one implementation instead of
two -- same pattern as phase1/pipeline.py's `run_*` functions.
"""

import difflib
import re
from collections import Counter

from sentence_transformers import SentenceTransformer

from common import ollama_client
from phase2.condensation_report import parse_condensed_blocks

FUNCTION_WORD_SET = {
    "a", "an", "the", "this", "that", "these", "those",
    "and", "or", "but", "if", "then", "so", "nor",
    "of", "in", "on", "at", "by", "for", "with", "without", "to", "from",
    "as", "is", "are", "was", "were", "be", "been", "being",
    "it", "its", "we", "they", "he", "she", "which", "who", "what",
    "i", "you", "your", "my", "our", "one",
    "will", "would", "can", "could", "shall", "should", "may", "might", "must",
}

# Nouns that are technically content-bearing by part of speech but function
# as pure discourse connectives in phrases like "by contrast", "in sum",
# "in short", "in turn". classify_span's F rule is POS/dependency-based
# now (see CONNECTIVE_POS/_is_connective_token below) and no longer reads
# this list directly; it's still used by flag_borderline_classifications'
# lexical cross-check, and by classify_span's own R/C content-word overlap
# calculation, where "contrast" etc. should still count normally there.
DISCOURSE_MARKER_EXTRAS = {"contrast", "sum", "short", "turn", "addition", "regard", "view"}

# Sentence-adverb discourse connectives (however/furthermore/therefore/
# thus/moreover/...) that _is_connective_token treats as connective via
# dep_=="advmod" or freestanding ADV+ROOT, but were never part of
# FUNCTION_WORD_SET (a much older, narrower list). Exempted here too so
# flag_borderline_classifications' lexical cross-check on F spans doesn't
# spuriously fire on every ordinary case -- these words are the whole
# point of that part of the connective test, not a disagreement to flag.
DISCOURSE_ADVERBS = {
    "furthermore", "however", "therefore", "thus", "moreover", "hence",
    "consequently", "nevertheless", "nonetheless", "meanwhile", "indeed",
    "accordingly", "additionally", "besides", "similarly", "likewise",
    "conversely", "otherwise", "instead", "still", "yet",
    # Latin discourse abbreviations -- spaCy tags these the same way
    # (dep_=="advmod", e.g. "i.e." is pos_='X'/tag_='FW'), so
    # _is_connective_token already treats them as connective; stored
    # here punctuation-stripped since that's what _content_words's
    # membership test actually compares against ("i.e." -> "ie").
    # Verified case: real Boisseau text "expertise -- i.e. a sense
    # that..." was flagged borderline solely because "ie" wasn't
    # exempted anywhere, not because of a genuine disagreement.
    "ie", "eg", "cf", "viz",
}

METALINGUISTIC_MARKERS = (
    "this paper", "this section", "this study", "this article", "this chapter",
    "the paper", "the article", "the following section", "the following sections",
    "these considerations", "this argument", "the argument", "as we have seen",
    "in what follows", "we now turn", "we conclude", "to summarize", "in sum",
    "this discussion", "the discussion above", "the preceding", "the remainder of",
)


# ============================================================
# GATHERING INPUTS FROM PHASE 1 / PHASE 2
# ============================================================

def gather_ordered_informative_sentences(phase1_state):
    """Flattens every cluster's most_insightful_sentences, deduped and
    sorted by corpus sentence index, so the LLM sees them in corpus order."""
    seen = {}
    for cluster in phase1_state.get("clusterDefs", []):
        for item in cluster.get("most_insightful_sentences", []):
            seen[item["index"]] = item
    return [seen[idx] for idx in sorted(seen)]


def gather_cluster_key_terms(phase1_state):
    return {
        cluster["name"]: list(cluster.get("stems", [])) + list(cluster.get("ngrams", []))
        for cluster in phase1_state.get("clusterDefs", [])
    }


# ============================================================
# CONDENSATION GENERATION
# ============================================================

def count_words(text):
    return len(text.split())


CTX_BUCKETS = (2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144)


def estimate_num_ctx(prompt, target_words, buckets=CTX_BUCKETS):
    """Rounds the prompt + expected-output size up to the next context
    bucket, instead of letting Ollama fall back to the model's own max
    context length (which can be far larger than needed and force the
    KV cache off the GPU -- see ollama_client.generate)."""
    estimated_tokens = (count_words(prompt) + target_words) * 1.4
    for bucket in buckets:
        if estimated_tokens <= bucket:
            return bucket
    return buckets[-1]


def target_word_count(source_text, rate_pct):
    return max(1, round(count_words(source_text) * rate_pct / 100))


def build_condensation_prompt(ordered_sentences, cluster_key_terms, target_words,
                               corpus_name, source_text=None):
    sentences_block = "\n".join(f"- {s['sentence']}" for s in ordered_sentences)
    terms_block = "\n".join(
        f"- {name}: " + ", ".join(terms[:15])
        for name, terms in cluster_key_terms.items() if terms
    )

    source_block = ""
    if source_text is not None:
        source_block = (
            "\n\nFor additional context, here is the full source text. Use it to "
            "faithfully represent the argument, but keep prioritizing the informative "
            "sentences and key terms above when deciding what to include:\n\n"
            f"{source_text}\n"
        )

    return (
        f'You are condensing an excerpt from the corpus "{corpus_name}" to '
        f"approximately {target_words} words (+/-5%).\n\n"
        "Below are the corpus's most informative sentences, in their original order, "
        "and the key terms of each semantic cluster identified in prior analysis. "
        "Write a single, coherent, readable condensation that:\n"
        "1. Follows the argumentative structure implied by these sentences, not "
        "necessarily their literal order.\n"
        "2. Faithfully represents the source's argument, structure, and conclusions.\n"
        f"3. Reaches approximately {target_words} words.\n"
        "4. Uses the source's own vocabulary and key terms wherever possible, "
        "especially the cluster terms listed below, but is not constrained to do so.\n"
        "5. Does not introduce examples, framings, or conclusions absent from the "
        "material below.\n\n"
        f"Informative sentences (in corpus order):\n{sentences_block}\n\n"
        f"Cluster key terms to try to preserve:\n{terms_block}"
        f"{source_block}\n\n"
        "Return only the condensed text, with no preamble or commentary."
    )


# ============================================================
# GENERATION SANITY CHECKS
# ============================================================
# Three simple, cheap heuristics that catch a generation gone visibly
# wrong before it's accepted -- not a coherence/quality judge (that would
# need an LLM call of its own), just a guard against clear degeneration
# failure modes local models can hit: topic drift/hallucinated vocabulary,
# runaway token concatenation, and verbatim repetition loops.

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _split_sentences_loose(text):
    """Lightweight regex sentence split -- condense.py has no spaCy
    dependency today, and pulling one in just for a rough per-sentence
    average would be a heavier dependency than this heuristic needs."""
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


def _check_invented_token_run(condensed_text, ordered_sentences):
    """Flags a run of consecutive words, at least as long as the
    condensation's own mean words-per-sentence, where every word is
    absent from the informative sentences actually fed into the prompt
    (not the full source text, which would contain virtually every word
    and defeat the check) and isn't an ordinary function word/connective
    (FUNCTION_WORD_SET, reused from the F/T classifier below) -- a sign
    of topic drift or hallucinated vocabulary rather than a paraphrase of
    the material the model was asked to use."""
    source_vocab = {
        w.lower() for s in ordered_sentences
        for w in re.findall(r"[A-Za-z']+", s["sentence"])
    }
    sentences = _split_sentences_loose(condensed_text)
    words = re.findall(r"[A-Za-z']+", condensed_text)
    if not sentences or not words:
        return None
    threshold = max(1, round(len(words) / len(sentences)))

    run, longest_run = [], []
    for w in words:
        lw = w.lower()
        if lw not in source_vocab and lw not in FUNCTION_WORD_SET:
            run.append(w)
            if len(run) > len(longest_run):
                longest_run = run[:]
        else:
            run = []

    if len(longest_run) >= threshold:
        preview = " ".join(longest_run[:12]) + ("..." if len(longest_run) > 12 else "")
        return (f"{len(longest_run)}-word run of terms absent from any informative "
                f"sentence (>= this condensation's {threshold}-word mean sentence length): {preview!r}")
    return None


def _check_mega_long_word(condensed_text, max_word_length=30):
    """Flags a single alphabetic token longer than max_word_length -- 30
    comfortably clears real English (even "antidisestablishmentarianism"
    is 28 characters) while catching runaway concatenated-word
    degeneration (e.g. "...pentakahectohexacontaheptaogontakaidoeka...")."""
    words = re.findall(r"[A-Za-z']+", condensed_text)
    longest = max(words, key=len, default="")
    if len(longest) > max_word_length:
        preview = longest[:50] + ("..." if len(longest) > 50 else "")
        return f"{len(longest)}-character word (over the {max_word_length}-character threshold): {preview!r}"
    return None


def _check_repeated_ngram(condensed_text, n=6, min_repeats=2):
    """Flags an exact n-word phrase that repeats at least min_repeats
    times anywhere in the text. Six consecutive words repeating verbatim
    is effectively impossible in coherent, non-quoting prose -- a strong,
    low-false-positive signal of a repetition-loop degeneration."""
    words = [w.lower() for w in re.findall(r"[A-Za-z']+", condensed_text)]
    if len(words) < n * min_repeats:
        return None

    counts = Counter(tuple(words[i:i + n]) for i in range(len(words) - n + 1))
    repeated = [(gram, count) for gram, count in counts.items() if count >= min_repeats]
    if not repeated:
        return None

    gram, count = max(repeated, key=lambda item: item[1])
    return f'{n}-word phrase repeated {count} times: "{" ".join(gram)}"'


def check_condensation_sanity(condensed_text, ordered_sentences):
    """Runs all three sanity checks; returns a list of human-readable
    issue descriptions (empty if the text looks clean)."""
    checks = (
        _check_invented_token_run(condensed_text, ordered_sentences),
        _check_mega_long_word(condensed_text),
        _check_repeated_ngram(condensed_text),
    )
    return [issue for issue in checks if issue]


def build_adversarial_review_prompt(condensed_text, ordered_sentences, target_words, corpus_name):
    """Prompt for the adversarial reviewer -- a second LLM pass over an
    already-accepted condensation, complementary to check_condensation_sanity's
    regex/statistical heuristics (invented-token runs, mega-long words,
    repeated n-grams): this pass instead makes a semantic/qualitative
    judgment call those heuristics can't (does it end mid-thought, does it
    drift outside the informative-sentence pool)."""
    sentences_block = "\n".join(f"- {s['sentence']}" for s in ordered_sentences)
    return (
        f'You are auditing a condensation of the corpus "{corpus_name}", '
        f"which was generated to target approximately {target_words} words.\n\n"
        "Below is the pool of informative sentences the condensation was supposed "
        "to be built from, followed by the condensation itself. Check the "
        "condensation for three specific defects:\n"
        "1. Does it end abruptly, mid-sentence or mid-thought?\n"
        "2. Does it contain garbled, repeated, or nonsensical sequences of words "
        "or tokens?\n"
        "3. Does its content drift substantially outside the topics/claims covered "
        "by the informative sentences below (i.e. does it introduce ideas not "
        "grounded in that pool)?\n\n"
        f"Informative sentences:\n{sentences_block}\n\n"
        f"Condensation to audit:\n{condensed_text}\n\n"
        "If none of these three defects are present, respond with exactly:\n"
        "VERDICT: OK\n\n"
        "If any defect is present, fix ONLY that defect (make the smallest "
        "possible edit; do not rewrite for style), keep the result as close as "
        f"possible to {target_words} words, and respond in exactly this format:\n"
        "VERDICT: FIXED\n"
        "ISSUES: <comma-separated short description of each defect found>\n"
        "TEXT:\n<the corrected condensation, and nothing else after it>"
    )


def parse_adversarial_response(response_text):
    """Parses the adversarial reviewer's structured response. Returns
    (verdict, issues, fixed_text): verdict is "OK", "FIXED", or
    "UNPARSEABLE" (the model didn't follow the format -- treated as a
    no-op by the caller, never as license to guess); issues is a list of
    short strings; fixed_text is the corrected condensation, or None."""
    match = re.search(r"VERDICT:\s*(OK|FIXED)", response_text, re.IGNORECASE)
    if not match:
        return "UNPARSEABLE", ["could not parse a VERDICT from the reviewer's response"], None

    verdict = match.group(1).upper()
    if verdict == "OK":
        return "OK", [], None

    issues_match = re.search(r"ISSUES:\s*(.+?)(?:\n\s*TEXT:|\Z)", response_text, re.IGNORECASE | re.DOTALL)
    issues = [i.strip() for i in issues_match.group(1).split(",") if i.strip()] if issues_match else []

    text_match = re.search(r"TEXT:\s*\n?(.*)\Z", response_text, re.IGNORECASE | re.DOTALL)
    fixed_text = text_match.group(1).strip() if text_match else None

    return "FIXED", issues, fixed_text


def run_adversarial_review(condensed_text, ordered_sentences, target_words, corpus_name,
                            adversarial_model, decisions, rate, host=ollama_client.DEFAULT_HOST,
                            timeout=600):
    """Runs the adversarial reviewer once over an already-accepted
    condensation. Returns (final_text, was_fixed, original_text, issues).

    Fails safe on any problem (unreachable Ollama, unparseable response, or
    a degenerate "fixed" text) by keeping the original text -- an
    adversarial-review failure should never be able to break the pipeline
    or silently replace good text with garbage. Every outcome is logged via
    the existing DecisionLog mechanism, with the original text always
    preserved in `extra` for audit even when a fix is applied."""
    prompt = build_adversarial_review_prompt(condensed_text, ordered_sentences, target_words, corpus_name)

    try:
        response = ollama_client.generate(adversarial_model, prompt, host=host, timeout=timeout, think=False)
    except ollama_client.OllamaError as e:
        decisions.record(
            step="condensation_generation", decision_type="adversarial_review_result",
            prompt="Adversarial reviewer check on accepted condensation",
            choice="unchanged",
            extra={"rate": rate, "model": adversarial_model, "issues_found": [f"reviewer call failed: {e}"],
                   "original_text": condensed_text, "fixed_text": None},
        )
        return condensed_text, False, condensed_text, [f"reviewer call failed: {e}"]

    verdict, issues, fixed_text = parse_adversarial_response(response)

    was_fixed = False
    final_text = condensed_text
    if verdict == "FIXED" and fixed_text and count_words(fixed_text) >= 0.5 * target_words:
        final_text = fixed_text
        was_fixed = True
    elif verdict == "FIXED":
        issues = issues + ["fixed text rejected as degenerate (empty or far too short); kept original"]

    decisions.record(
        step="condensation_generation", decision_type="adversarial_review_result",
        prompt="Adversarial reviewer check on accepted condensation",
        choice="fixed" if was_fixed else "unchanged",
        extra={"rate": rate, "model": adversarial_model, "issues_found": issues,
               "original_text": condensed_text, "fixed_text": final_text if was_fixed else None},
    )
    return final_text, was_fixed, condensed_text, issues


def format_trial_line(t, prefix=""):
    """One console/log line per generation trial -- word-count result plus
    any sanity issues, if present. Shared (not a leading-underscore
    helper) so run_pipeline.py and webapp/pipeline_session.py's own
    trial-printing loops render the exact same format as this module's,
    rather than re-deriving it three more times."""
    status = "OK" if t["within_tolerance"] else "outside tolerance"
    line = f"  {prefix}trial {t['trial']}: {t['word_count']} words (target {t['target_words']}, {status})"
    if t.get("sanity_issues"):
        line += f" -- {len(t['sanity_issues'])} sanity issue(s): " + "; ".join(t["sanity_issues"])
    return line


def attempt_condensation_trials(ordered_sentences, cluster_key_terms, target_words, corpus_name,
                                 model, max_trials, include_full_text, source_text=None,
                                 host=ollama_client.DEFAULT_HOST, tolerance=0.05, timeout=900,
                                 prompt_override=None):
    """Runs up to max_trials generation attempts, no input() involved.
    Returns {"text": str|None, "trials": [...], "success": bool,
    "soft_accept": bool, "sanity_failed": bool, "sanity_review_candidates":
    [...]}. The caller decides what to do next (offer a sanity-check
    review, retry escalated, give up).

    Every trial is checked both for word-count tolerance and for basic
    generation sanity (check_condensation_sanity -- catches topic drift,
    runaway token concatenation, and repetition loops). A trial is only
    the immediate "success" return when it clears both.

    A trial that hits the word-count target but fails the sanity check is
    NOT auto-selected either way -- the sanity check is a heuristic
    (verbatim-repetition detector etc.), not a correctness guarantee, so
    silently keeping or silently discarding such a trial both risk being
    wrong. Every trial in that situation across the run is collected into
    "sanity_review_candidates" (word count + which sanity check(s) it
    failed, no auto-picked winner) for the caller to put in front of the
    researcher. "text" is left None and "success" False whenever any such
    candidates exist, even if a full success or a soft-accept fallback
    would otherwise have been available -- the point is that this
    specific situation always goes to the researcher, not that the
    pipeline degrades to guessing for it.

    Failing that, falls back the same way it always has: closest trial by
    word count among the ones that passed the sanity check ("soft_accept":
    True) rather than discarding everything -- an over-length-but-coherent
    response still produces something usable instead of being thrown
    away. Only if every trial failed the sanity check does the fallback
    fall back further, to the closest trial overall, tagged
    "sanity_failed": True so callers/logs can disclose that clearly --
    never silently discard output, same principle as soft_accept itself.
    "success" stays False whenever tolerance wasn't hit, so callers can
    still offer escalation; only a genuinely empty result (text is None
    and sanity_review_candidates is empty) means every trial failed
    outright (e.g. max_trials == 0).

    `prompt_override`, if given, is used verbatim as the prompt for every
    trial instead of the one build_condensation_prompt would render --
    used by the post-condensation "regenerate" flow, where the researcher
    has freely edited the rendered default prompt (or left it untouched).

    `timeout` (seconds) is passed straight to ollama_client.generate --
    large "thinking"-capable local models can take well over the default
    600s Ollama-client timeout to respond, so this is worth raising for
    slow models rather than treating a timeout as a hard failure."""
    trials = []
    best_diff, best_text = None, None
    fallback_diff, fallback_text = None, None
    sanity_review_candidates = []
    for trial in range(1, max_trials + 1):
        prompt = prompt_override if prompt_override is not None else build_condensation_prompt(
            ordered_sentences, cluster_key_terms, target_words, corpus_name,
            source_text=source_text if include_full_text else None,
        )
        num_ctx = estimate_num_ctx(prompt, target_words)
        # repeat_penalty=1.0 (Ollama/llama.cpp's neutral value) overrides
        # whatever non-1.0 default the model's own Modelfile applies, so
        # no implicit anti-repetition bias suppresses legitimate reuse of
        # source/cluster key terms the condensation is supposed to preserve.
        text = ollama_client.generate(model, prompt, host=host, timeout=timeout,
                                       options={"num_ctx": num_ctx, "repeat_penalty": 1.0}, think=False)
        word_count = count_words(text)
        diff = abs(word_count - target_words)
        within_tolerance = diff <= tolerance * target_words
        sanity_issues = check_condensation_sanity(text, ordered_sentences)
        sane = not sanity_issues

        trials.append({
            "trial": trial,
            "word_count": word_count,
            "target_words": target_words,
            "within_tolerance": within_tolerance,
            "sanity_issues": sanity_issues,
        })

        if within_tolerance and sane:
            return {"text": text, "trials": trials, "success": True, "soft_accept": False,
                    "sanity_failed": False, "sanity_review_candidates": []}

        if within_tolerance and not sane:
            sanity_review_candidates.append({
                "trial": trial, "word_count": word_count, "target_words": target_words,
                "text": text, "sanity_issues": sanity_issues,
            })

        if sane and (best_diff is None or diff < best_diff):
            best_diff, best_text = diff, text
        if not sane and (fallback_diff is None or diff < fallback_diff):
            fallback_diff, fallback_text = diff, text

    if sanity_review_candidates:
        return {"text": None, "trials": trials, "success": False, "soft_accept": False,
                "sanity_failed": False, "sanity_review_candidates": sanity_review_candidates}
    if best_text is not None:
        return {"text": best_text, "trials": trials, "success": False, "soft_accept": True,
                "sanity_failed": False, "sanity_review_candidates": []}
    if fallback_text is not None:
        return {"text": fallback_text, "trials": trials, "success": False, "soft_accept": True,
                "sanity_failed": True, "sanity_review_candidates": []}
    return {"text": None, "trials": trials, "success": False, "soft_accept": False,
            "sanity_failed": False, "sanity_review_candidates": []}


# ============================================================
# SOURCE TITLE / AUTHOR(S) / DATE
# ============================================================
# The condensation report needs the source document's own title/author/
# date for its header -- deliberately NOT guessed from the condensation
# itself (parse_condensed_blocks used to do that positionally and got it
# wrong whenever the condensation didn't open with a literal one-line
# title). Collected once per corpus run: either typed in directly, or,
# if left to the model, extracted via a small retrieval-augmented lookup
# over the source text (chunk -> embed -> retrieve top_k -> ask the LLM).

def _chunk_source_text(source_text, max_chunk_words=120):
    """Paragraph-level chunks (blank-line boundaries) -- title/author/date
    metadata is normally a short front-matter block, so paragraph-sized
    chunks preserve more of that context than sentence-level chunks would.
    Any paragraph longer than max_chunk_words is further split so no
    single chunk dominates the embedding/prompt."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", source_text) if p.strip()]
    chunks = []
    for p in paragraphs:
        words = p.split()
        if not words:
            continue
        for i in range(0, len(words), max_chunk_words):
            chunks.append(" ".join(words[i:i + max_chunk_words]))
    return chunks


def _parse_source_metadata_response(response):
    """Parses the fixed "TITLE: ...\\nAUTHOR(S): ...\\nDATE: ..." format
    back into (title, authors, date) -- plain-text, not JSON, matching how
    this codebase already treats local-model output elsewhere (avoids
    relying on strict JSON compliance from small/local models). Any field
    the response doesn't include, or leaves blank, is "Unclear"."""
    title = authors = date = "Unclear"
    for line in response.splitlines():
        line = line.strip()
        if line.upper().startswith("TITLE:"):
            title = line.split(":", 1)[1].strip() or "Unclear"
        elif line.upper().startswith("AUTHOR"):
            authors = line.split(":", 1)[1].strip() or "Unclear"
        elif line.upper().startswith("DATE:"):
            date = line.split(":", 1)[1].strip() or "Unclear"
    return title, authors, date


def extract_source_metadata(source_text, model, embedder_name="all-MiniLM-L6-v2",
                             top_k=5, host=ollama_client.DEFAULT_HOST, lead_chunks=3):
    """Retrieval-augmented title/author(s)/date extraction: embeds the
    source's paragraph-level chunks plus one synthetic query describing
    what we're looking for, retrieves the top_k most relevant chunks by
    cosine similarity, and asks the LLM to extract structured metadata
    from just those chunks -- not the whole source, which for a
    front-matter-only task is usually far more than a local model needs
    and dilutes its attention. Returns (title, authors, date); any field
    the model isn't confident about comes back "Unclear" (an explicit
    instruction in the prompt, not a guess).

    Pure semantic retrieval is unreliable for this specific task: a title
    or a bare author name is a short, information-sparse chunk that often
    scores *worse* against the query than an unrelated body paragraph
    happens to (verified empirically -- on a real corpus, the title and
    author chunks ranked 159th and 124th of 181 by cosine similarity,
    nowhere near top_k). Front matter is, however, reliably positioned at
    the very start of virtually every document, so the first lead_chunks
    chunks are always included alongside the semantically retrieved ones
    rather than relying on embedding similarity to surface them."""
    chunks = _chunk_source_text(source_text)
    if not chunks:
        return "Unclear", "Unclear", "Unclear"

    embedder = SentenceTransformer(embedder_name)
    query = "The document's title, author(s) or byline, and publication date."
    chunk_embeddings = embedder.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
    query_embedding = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    scores = chunk_embeddings @ query_embedding
    top_indices = set(scores.argsort()[::-1][:top_k].tolist())
    top_indices.update(range(min(lead_chunks, len(chunks))))
    retrieved = "\n\n---\n\n".join(chunks[i] for i in sorted(top_indices))

    prompt = (
        "Below are excerpts from a document, retrieved because they are most likely "
        "to contain its title, author(s), and publication date.\n\n"
        f"{retrieved}\n\n"
        "Based only on these excerpts, respond in exactly this format (one line each, "
        "nothing else):\n"
        "TITLE: <the document's title, or Unclear>\n"
        "AUTHOR(S): <the author(s)/byline, or Unclear>\n"
        "DATE: <the publication date, or Unclear>\n"
        "If you are not reasonably confident about a field, write Unclear for it -- do not guess."
    )
    response = ollama_client.generate(model, prompt, host=host, think=False)
    return _parse_source_metadata_response(response)


def run_source_metadata_setup(decisions, source_text, ollama_model, host=ollama_client.DEFAULT_HOST):
    """Interactive: collects the source's title/author(s)/date for the
    report header -- typed in directly, or left to the model (extracted
    via extract_source_metadata). Every choice is appended to the
    decisions log via `decisions.record(...)` for later audit. Shared by
    phase2.ipynb and run_pipeline.py so there's one implementation, not
    two. Returns (title, authors, date)."""
    mode = input(
        "\nEnter the source's title, author(s), and date for the report header? "
        "(y = enter manually / n = let the LLM determine them): "
    ).strip().lower()

    decisions.record(
        step="source_metadata", decision_type="mode_choice",
        prompt="Enter title/author(s)/date manually, or let the LLM determine them?",
        options=["y", "n"], choice=mode,
    )

    if mode == "y":
        title = input("Title: ").strip() or "Unclear"
        authors = input("Author(s): ").strip() or "Unclear"
        date = input("Date: ").strip() or "Unclear"
    else:
        print("\nExtracting title/author(s)/date from the source text...")
        title, authors, date = extract_source_metadata(source_text, ollama_model, host=host)
        print(f"  Title: {title}\n  Author(s): {authors}\n  Date: {date}")

    decisions.record(
        step="source_metadata", decision_type="metadata_result",
        prompt="Source title/author(s)/date", choice=mode,
        extra={"title": title, "authors": authors, "date": date},
    )

    return title, authors, date


def run_condensation_setup(decisions, host=ollama_client.DEFAULT_HOST):
    """Interactive condensation setup: Ollama availability check plus the
    rate(s)/model/max-trials prompts. Every choice is appended to the
    decisions log via `decisions.record(...)` for later audit -- it does
    not change the interactive flow. Shared by phase2.ipynb and
    run_pipeline.py so there's one implementation, not two. Returns
    (rates, ollama_model, max_trials, adversarial_model)."""
    if not ollama_client.is_available(host=host):
        print(
            f"Ollama does not appear to be running at {host}.\n"
            "Install it from https://ollama.com, pull a model (e.g. `ollama pull llama3`), "
            "and make sure `ollama serve` is running, then re-run this cell."
        )
    else:
        print("Ollama is available. Installed models:")
        for name in ollama_client.list_models(host=host):
            print(f"  - {name}")

    rate_input = input(
        "\nWhat condensation rate(s) would you like? (5-30%, comma-separated for multiple, e.g. '10,20'): "
    ).strip()
    rates = [int(r.strip()) for r in rate_input.split(",") if r.strip()]

    decisions.record(
        step="condensation_setup", decision_type="rate_selection",
        prompt="What condensation rate(s) would you like? (5-30%)",
        choice=rate_input, extra={"parsed_rates": rates},
    )

    ollama_model = input("\nWhich Ollama model would you like to use? ").strip()

    decisions.record(
        step="condensation_setup", decision_type="ollama_model_choice",
        prompt="Which Ollama model would you like to use?", choice=ollama_model,
    )

    adversarial_model = input(
        f"\nWhich Ollama model should the adversarial reviewer use? "
        f"(ENTER to reuse '{ollama_model}'): "
    ).strip() or ollama_model

    decisions.record(
        step="condensation_setup", decision_type="adversarial_model_choice",
        prompt="Which Ollama model should the adversarial reviewer use?", choice=adversarial_model,
    )

    max_trials = int(input("\nHow many generation trials before asking to escalate? ").strip())

    decisions.record(
        step="condensation_setup", decision_type="max_trials_choice",
        prompt="How many generation trials before asking to escalate?", choice=str(max_trials),
    )

    print(
        f"\nWill generate condensations at {rates}% using '{ollama_model}', "
        f"up to {max_trials} trial(s) each before escalation. Adversarial reviewer: "
        f"'{adversarial_model}'."
    )

    return rates, ollama_model, max_trials, adversarial_model


def run_sanity_review_for_rate(rate, result, decisions, escalated=False):
    """Interactive: at least one trial for this rate hit the target word
    count but failed a generation sanity check -- attempt_condensation_trials
    never auto-picks in that situation (see its docstring), since the
    sanity check is a heuristic (verbatim-repetition detector etc.), not
    a confirmed error, and silently keeping or silently discarding it are
    both risky. Prints every such candidate with its word count and the
    specific check(s) it failed, and lets the researcher accept one of
    them or reject all of them. Rejecting leaves result["text"] None so
    the normal give-up/escalation path runs next, same as if nothing had
    hit tolerance at all. Every choice is appended to the decisions log
    via `decisions.record(...)`. Shared by phase2.ipynb and
    run_pipeline.py. Mutates and returns `result`."""
    candidates = result["sanity_review_candidates"]
    label = "escalated " if escalated else ""
    print(
        f"\n{len(candidates)} {label}trial(s) at {rate}% hit the target word count but "
        "failed a generation sanity check (a heuristic, not a confirmed error) -- "
        "review and pick one to keep, or reject all of them:"
    )
    for c in candidates:
        print(f"\n  [{c['trial']}] {c['word_count']} words (target {c['target_words']})")
        for issue in c["sanity_issues"]:
            print(f"      - {issue}")

    choice = input(
        f"\nAccept one of these trials for {rate}%? Enter a trial number, or ENTER for none: "
    ).strip()

    accepted = next((c for c in candidates if str(c["trial"]) == choice), None) if choice else None
    if choice and accepted is None:
        print(f"'{choice}' isn't one of the listed trial numbers -- treating as none.")

    decisions.record(
        step="condensation_generation", decision_type="sanity_review_choice",
        prompt="Accept a trial that hit the word-count target but failed a sanity check?",
        choice=str(accepted["trial"]) if accepted else "none",
        extra={"rate": rate, "escalated": escalated,
               "candidates": [{"trial": c["trial"], "word_count": c["word_count"],
                                "sanity_issues": c["sanity_issues"]} for c in candidates]},
    )

    if accepted is not None:
        result["text"] = accepted["text"]
        result["researcher_accepted_despite_sanity"] = True
        print(f"\nAccepted trial {accepted['trial']} for {rate}% despite the sanity flag above.")
    else:
        result["text"] = None
        print(f"\nRejected all sanity-flagged trials for {rate}%.")

    return result


def run_generate_for_rate(rate, ordered_sentences, cluster_key_terms, target_words, corpus_name,
                           text, model, max_trials, decisions, host=ollama_client.DEFAULT_HOST):
    """Runs the full interactive generate-then-maybe-escalate flow for one
    rate: non-escalated trials, then (if any hit the word-count target
    but failed a sanity check) a sanity review, then (if still nothing
    accepted) asks whether to retry with the full source text included --
    logging every trial and choice via `decisions.record(...)`. Saving
    the resulting text to disk stays with the caller. Shared by
    phase2.ipynb and run_pipeline.py so there's one implementation, not
    two. Returns the same {"text", "trials", "success"} shape as
    attempt_condensation_trials, plus "researcher_accepted_despite_sanity"
    when a sanity-flagged trial was explicitly kept."""
    print(f"\n=== {rate}% condensation (target ~{target_words} words) ===")

    result = attempt_condensation_trials(
        ordered_sentences, cluster_key_terms, target_words, corpus_name,
        model=model, max_trials=max_trials, include_full_text=False, host=host,
    )

    for t in result["trials"]:
        print(format_trial_line(t))
        decisions.record(
            step="condensation_generation", decision_type="trial_result",
            prompt="Condensation generation trial",
            extra={"rate": rate, **t},
        )

    if result["sanity_review_candidates"]:
        result = run_sanity_review_for_rate(rate, result, decisions)

    if not result["success"] and not result.get("researcher_accepted_despite_sanity"):

        escalate = input(
            f"\n{max_trials} trial(s) at {rate}% did not produce an accepted condensation.\n"
            "Retry with the full source text included in the prompt? (y/n): "
        ).strip().lower()

        decisions.record(
            step="condensation_generation", decision_type="escalation_prompt",
            prompt="Retry with the full source text included in the prompt?",
            options=["y", "n"], choice=escalate, extra={"rate": rate},
        )

        if escalate == "y":

            result = attempt_condensation_trials(
                ordered_sentences, cluster_key_terms, target_words, corpus_name,
                model=model, max_trials=max_trials, include_full_text=True,
                source_text=text, host=host,
            )

            for t in result["trials"]:
                print(format_trial_line(t, prefix="[escalated] "))
                decisions.record(
                    step="condensation_generation", decision_type="trial_result",
                    prompt="Condensation generation trial (escalated)",
                    extra={"rate": rate, "escalated": True, **t},
                )

            if result["sanity_review_candidates"]:
                result = run_sanity_review_for_rate(rate, result, decisions, escalated=True)

    if result["text"] is None:
        print(f"\nGiving up on {rate}% -- no condensation accepted. Skipping verification/export for this rate.")
    elif result.get("researcher_accepted_despite_sanity"):
        pass  # already reported by run_sanity_review_for_rate above
    elif result.get("sanity_failed"):
        print(
            f"\n{rate}%: no trial passed the sanity checks -- using the closest trial by word count anyway "
            f"({count_words(result['text'])} words vs. target {target_words}) rather than discarding it, "
            "but flag it for a manual look."
        )
    elif result.get("soft_accept"):
        print(
            f"\n{rate}%: no trial hit the target word count within tolerance -- "
            f"using the closest trial ({count_words(result['text'])} words vs. target {target_words}) "
            "instead of discarding it."
        )

    return result


# ============================================================
# VERBATIM-OVERLAP SCAN
# ============================================================

def in_source(chunk_words, source_lower):
    phrase = " ".join(w.lower() for w in chunk_words)
    phrase_clean = re.sub(r"[^\w\s]", "", phrase)
    src_clean = re.sub(r"[^\w\s]", "", source_lower)
    return phrase_clean in src_clean


def scan_verbatim_overlap(condensed_text, source_text, min_window=4, max_window=40):
    """For each position in the condensed text, finds the longest run of
    consecutive tokens (capped at max_window) that appears verbatim in the
    source. Returns (matched_token_count, total_token_count, matched_runs)
    -- matched_runs is a list of (start_idx, end_idx, tokens)."""
    source_lower = source_text.lower()
    cond_tokens = re.findall(r"[A-Za-z']+", condensed_text)

    i, matched_runs = 0, []
    while i < len(cond_tokens):
        best_len = 0
        for length in range(min(max_window, len(cond_tokens) - i), min_window - 1, -1):
            if in_source(cond_tokens[i:i + length], source_lower):
                best_len = length
                break
        if best_len >= min_window:
            matched_runs.append((i, i + best_len, cond_tokens[i:i + best_len]))
            i += best_len
        else:
            i += 1

    matched_token_count = sum(end - start for start, end, _ in matched_runs)
    return matched_token_count, len(cond_tokens), matched_runs


# ============================================================
# INJECTION CLASSIFICATION (F/T/R/C)
# ============================================================

def _content_words(text, extra_function_words=frozenset()):
    words = text.split()
    ignore = FUNCTION_WORD_SET | extra_function_words
    return [
        w for w in words
        if re.sub(r"[^a-zA-Z]", "", w).lower() not in ignore
        and re.sub(r"[^a-zA-Z]", "", w) != ""
    ]


def is_metalinguistic(sentence):
    lowered = sentence.lower()
    return any(marker in lowered for marker in METALINGUISTIC_MARKERS)


# F ("purely connective") is decided from spaCy POS/dependency tags rather
# than a fixed word list. CCONJ/SCONJ cover true conjunctions (and/but/or/
# because/although/since/while/that/though). dep_ == "advmod" catches
# sentence-adverb discourse connectives (however/furthermore/therefore/
# thus/moreover) when they attach to a governing verb elsewhere in the
# span -- but in a short standalone fragment with no verb to attach to
# (e.g. "Furthermore," or "But then," on their own), spaCy's parser makes
# the adverb the span's ROOT instead, so pos_ == "ADV" and dep_ == "ROOT"
# is also accepted. spaCy has no dedicated "discourse adverb" tag, so both
# checks reuse dependency labels ordinary manner/degree adverbs can get
# too ("quickly", "very"); accepted as an approximation since it's only
# ever evaluated on <=4-token spans, where a stray manner/degree adverb is
# unlikely. ADP (prepositions) is deliberately excluded: "in addition"/"by
# contrast" have a real content noun as their object, so they're not
# purely connective -- they fall through to the T safety net below instead.
CONNECTIVE_POS = {"CCONJ", "SCONJ"}


def _is_connective_token(token):
    if token.pos_ in CONNECTIVE_POS:
        return True
    if token.dep_ == "advmod":
        return True
    if token.pos_ == "ADV" and token.dep_ == "ROOT":
        return True
    return False


def _non_trivial_tokens(span):
    return [t for t in span if not t.is_punct and not t.is_space]


# T's "genuine lexical content" category set -- deliberately broad (not
# just noun/verb/adjective) per the same "generalist spaCy category"
# philosophy as the wh-word tag_ check below: NOUN/PROPN/VERB/ADJ cover
# the named examples, PRON/ADV/NUM catch other content-bearing insertions
# (a pronoun, a quantity) without needing a category per part of speech.
LEXICAL_POS = {"NOUN", "PROPN", "VERB", "ADJ", "PRON", "ADV", "NUM"}


def _is_lexical_token(token):
    """A token counts as genuine inserted/changed content for T's
    purposes -- but never a token _is_connective_token already claims for
    F. This matters because ADV overlaps both sets: an inserted adverb
    that's a discourse connective (dep_ == "advmod", F's job) must not
    also satisfy T's "lexical" test just because its coarse POS is ADV.
    F is always checked first and returns immediately on a match, so this
    exclusion is what keeps a >4-token all-connective diff (too long for
    F) from wrongly qualifying for T too -- it correctly falls through to
    R/C instead, since none of its tokens are "lexical" by this test."""
    if _is_connective_token(token):
        return False
    return token.pos_ in LEXICAL_POS or token.tag_.startswith("W")


def _diff_inserted_tokens(condensed_tokens, source_word_tokens):
    """condensed_tokens: spaCy tokens (punctuation/space already stripped,
    via _non_trivial_tokens). source_word_tokens: plain lowercased
    alpha-only strings from the best-matching source sentence. Returns the
    condensed_tokens that fall within an "insert" or "replace" diff
    opcode against source_word_tokens -- i.e. tokens genuinely new or
    changed relative to that source sentence. Pure deletions (source
    words simply absent from the condensation) introduce nothing new to
    classify and are ignored."""
    condensed_texts = [t.text.lower() for t in condensed_tokens]
    matcher = difflib.SequenceMatcher(None, source_word_tokens, condensed_texts, autojunk=False)
    diff_tokens = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("insert", "replace"):
            diff_tokens.extend(condensed_tokens[j1:j2])
    return diff_tokens


def classify_span(sent, source_sentences, source_lower, span_id):
    """sent: a spaCy Span for one sentence (as yielded by nlp(text).sents).
    Returns a span dict, or None if the sentence is verbatim in the source
    (the taxonomy only classifies non-verbatim spans).

    F and T are whole-sentence classifications now, not independent
    short-span rules: for every non-verbatim sentence, first find its
    best-matching source sentence (the same content-word-overlap method
    R/C already used, just run unconditionally/earlier), then diff the
    condensed sentence's tokens against that source sentence's tokens. If
    the entire delta is small and purely connective, the whole sentence is
    F -- it's essentially the source sentence plus a bit of connective
    glue. If the delta is small and contains genuine lexical content, the
    whole sentence is T. R keeps its existing overlap-based test, now only
    reached once F and T are ruled out; C is the unchanged fallback."""
    sentence = sent.text.strip()
    words = sentence.split()

    if in_source(words, source_lower):
        return None

    tokens = _non_trivial_tokens(sent)

    # Find the best-matching source sentence -- unchanged content-word
    # overlap method, just run for every span now (guarded so a span with
    # no content words at all -- e.g. a freestanding connective fragment
    # -- skips straight past it instead of dividing by zero).
    span_content = {w.lower() for w in _content_words(sentence)}
    best_idx, best_overlap, second_overlap = None, 0.0, 0.0
    overlap_scores = []  # every (idx, overlap) with overlap > 0 -- for C's multi-source attribution below
    if span_content:
        for idx, src_sentence in enumerate(source_sentences):
            src_content = {w.lower() for w in _content_words(src_sentence)}
            if not src_content:
                continue
            overlap = len(span_content & src_content) / len(span_content)
            if overlap > 0:
                overlap_scores.append((idx, overlap))
            if overlap > best_overlap:
                second_overlap = best_overlap
                best_overlap, best_idx = overlap, idx
            elif overlap > second_overlap:
                second_overlap = overlap

    diff_tokens = []
    if best_idx is not None:
        source_word_tokens = re.findall(r"[a-z']+", source_sentences[best_idx].lower())
        diff_tokens = _diff_inserted_tokens(tokens, source_word_tokens)

    # F: the sentence is otherwise identical to its best-matching source
    # sentence -- the only difference is <=4 purely connective tokens.
    if diff_tokens and len(diff_tokens) <= 4 and all(_is_connective_token(t) for t in diff_tokens):
        inserted = " ".join(t.text for t in diff_tokens)
        return {
            "span_id": span_id, "type": "F", "text": sentence,
            "source_refs": [best_idx],
            "justification": f"Diverges from source sentence {best_idx} only by inserting connective(s): {inserted!r}.",
            "basis": "diff_connective", "diff_text": [t.text for t in diff_tokens],
        }

    # No source sentence shared any content word at all (best_idx is
    # None) -- nothing to diff against. Falls back to the direct
    # descendant of the old standalone F rule, for a genuinely
    # freestanding connective fragment like "Furthermore," on its own.
    if best_idx is None and tokens and len(tokens) <= 4 and all(_is_connective_token(t) for t in tokens):
        return {
            "span_id": span_id, "type": "F", "text": sentence,
            "source_refs": [], "justification": "", "basis": "pure_connective_no_match",
        }

    if is_metalinguistic(sentence):
        # source_refs stays [] -- a metalinguistic span is meta-commentary
        # on the argument as a whole, not officially a paraphrase of any
        # one source sentence, so it doesn't get the R/C-style attribution.
        # But the content-word-overlap match above already ran (best_idx),
        # and this rule is a blunt substring test that can misfire on a
        # sentence that's mostly real object-level content (see
        # flag_borderline_classifications) -- candidate_source_ref keeps
        # that already-computed match around, unused unless this span
        # gets flagged borderline, so a reviewer isn't left with zero
        # context to judge the call. candidate_overlap/candidate_diff_text
        # carry forward the SAME diff_tokens already computed above (for
        # the F check) so flag_borderline_classifications can, when the
        # candidate is a strong match, flag based on how much this span
        # actually diverges from it instead of an absolute whole-sentence
        # word count -- diff_tokens is deliberately not thrown away here.
        return {
            "span_id": span_id, "type": "T", "text": sentence,
            "source_refs": [], "justification": "", "basis": "phrase_match",
            "candidate_source_ref": best_idx,
            "candidate_overlap": round(best_overlap, 4),
            "candidate_diff_text": [t.text for t in diff_tokens] if best_idx is not None else None,
        }

    # T: a small (<=6-token) delta from the best-matching source sentence
    # that contains genuine lexical content (not just connective glue).
    if diff_tokens and len(diff_tokens) <= 6 and any(_is_lexical_token(t) for t in diff_tokens):
        inserted = " ".join(t.text for t in diff_tokens)
        return {
            "span_id": span_id, "type": "T", "text": sentence,
            "source_refs": [best_idx],
            "justification": f"Diverges from source sentence {best_idx} by inserting non-connective content: {inserted!r}.",
            "basis": "diff_lexical", "diff_text": [t.text for t in diff_tokens],
        }

    # A single dominant match -> paraphrase of one sentence (R), only
    # reachable once F and T are both ruled out above.
    if best_idx is not None and best_overlap >= 0.5 and (best_overlap - second_overlap) >= 0.15:
        return {
            "span_id": span_id, "type": "R", "text": sentence,
            "source_refs": [best_idx],
            "justification": f"Paraphrases source sentence {best_idx} (content-word overlap {best_overlap:.0%}).",
        }

    # C: content drawn from/merged across multiple source sentences -- report
    # every source sentence with a meaningful (>=0.15) share of overlap, not
    # just the single best match, sorted strongest-first. A much lower bar
    # than R's 0.5 "dominant match" threshold on purpose: a contributing
    # source only needs to plausibly have fed into the compression, not
    # explain most of it on its own.
    contributing = [idx for idx, overlap in sorted(overlap_scores, key=lambda pair: pair[1], reverse=True)
                    if overlap >= 0.15]
    return {
        "span_id": span_id, "type": "C", "text": sentence,
        "source_refs": contributing,
        "justification": "Content does not trace cleanly to a single source sentence; treated as a compression of multiple.",
    }


def classify_condensation(condensed_text, source_text, nlp):
    """Classifies every non-verbatim sentence in the condensed text's body
    prose (p/defn blocks only -- headings and the title/author lines are
    structural, not argumentative content, and are skipped).

    Sentences are segmented block-by-block using the same
    `parse_condensed_blocks` the HTML renderer uses, rather than
    sentence-segmenting the raw text as a whole -- this guarantees every
    classified span's text is a substring of the exact block it will later
    be rendered into (render_spans matches spans to blocks by substring),
    and avoids spaCy merging an unpunctuated title line into the first
    body sentence.
    """
    source_lower = source_text.lower()
    source_sentences = [s.text.strip() for s in nlp(source_text).sents if s.text.strip()]

    all_spans = []
    span_counter = 0
    for block_type, text in parse_condensed_blocks(condensed_text):
        if block_type not in ("p", "defn"):
            continue
        for sent in nlp(text).sents:
            sentence = sent.text.strip()
            if not sentence:
                continue
            span_counter += 1
            span = classify_span(sent, source_sentences, source_lower, span_id=f"s{span_counter}")
            if span is not None:
                all_spans.append(span)

    return all_spans, source_sentences


def flag_borderline_classifications(all_spans):
    """Flags F/T spans that exceed their own type's stated definition, for
    researcher review -- not a confirmed error, a required disclosure item.
    Default is to keep the original classification unless the researcher
    requests reclassification (handled interactively in the notebook).

    classify_span's F/T decisions are POS/dependency-based now, not
    lexical -- the checks below re-run the old word-list heuristic as an
    independent cross-check and flag disagreement, rather than
    re-deriving a fixed density threshold:

    - "pure_connective_no_match" (the whole-sentence F fallback for spans
      with no source match to diff against): cross-checks the whole
      sentence, same as the original standalone F rule always did.
    - "diff_connective" (F): cross-checks only the diff_text tokens
      (what was actually inserted/changed) -- the sentence as a whole may
      legitimately contain lots of real content in its matched/aligned
      portion now, so flagging on the whole sentence would misfire on
      every ordinary case.
    - "phrase_match" (T, the original metalinguistic-phrase rule): if the
      span's candidate source sentence is a strong match (overlap >= 0.7 --
      deliberately stricter than R's own 0.5 "this candidate explains the
      sentence" bar, since a diff-based flag here is only trustworthy
      against a near-verbatim candidate, not merely a dominant one),
      cross-checks only the diff against that candidate -- same diff-based
      precision as diff_connective/
      diff_lexical, catching e.g. a metalinguistic-tagged sentence that's
      actually near-verbatim with one real word changed. Otherwise (no
      candidate, or too weak to trust a diff against) falls back to the
      original whole-sentence content-density check, since there's no
      well-defined "correct" source sentence to diff against.
    - "diff_lexical" (T, the new diff-based rule): never flagged --
      containing lexical content is expected by construction, not a
      borderline signal."""
    flags = []
    for s in all_spans:
        if s["type"] not in ("F", "T"):
            continue
        basis = s.get("basis")

        if s["type"] == "F" and basis == "pure_connective_no_match":
            words = s["text"].split()
            content_words = _content_words(s["text"], DISCOURSE_MARKER_EXTRAS | DISCOURSE_ADVERBS)
            if len(words) > 4 or len(content_words) > 0:
                flags.append({
                    "span_id": s["span_id"], "type": "F", "text": s["text"],
                    "source_refs": s.get("source_refs", []),
                    "reason": (f"{len(words)} words, {len(content_words)} lexically content-bearing "
                               "token(s) -- the POS/dependency classifier called this a pure "
                               "connective/discourse-adverb fragment (<=4 tokens); the word-list "
                               "heuristic disagrees, worth a manual check."),
                })
        elif s["type"] == "F" and basis == "diff_connective":
            diff_text = " ".join(s.get("diff_text", []))
            content_words = _content_words(diff_text, DISCOURSE_MARKER_EXTRAS | DISCOURSE_ADVERBS)
            if content_words:
                flags.append({
                    "span_id": s["span_id"], "type": "F", "text": s["text"],
                    "source_refs": s.get("source_refs", []),
                    "reason": (f"Inserted token(s) {diff_text!r} were called purely connective by the "
                               f"POS/dependency classifier, but the word-list heuristic considers "
                               f"{len(content_words)} of them content-bearing -- worth a manual check."),
                })
        elif s["type"] == "T" and basis == "phrase_match":
            # This span's real source_refs is always [] by design (see
            # classify_span) -- but candidate_source_ref/candidate_overlap/
            # candidate_diff_text kept the content-word-overlap match (and
            # its diff) that was computed anyway, so a reviewer deciding
            # whether to move this to R/C can see what it would most
            # likely attribute to instead of nothing. Not an official
            # match -- source_is_candidate_only tells callers to label it
            # as such, not as a confirmed attribution.
            candidate_ref = s.get("candidate_source_ref")
            candidate_overlap = s.get("candidate_overlap") or 0.0
            candidate_diff_text = s.get("candidate_diff_text")

            if candidate_diff_text is not None and candidate_overlap >= 0.7:
                # Strong candidate match -- flag based on how much this
                # span actually diverges from it, not an absolute
                # whole-sentence word count (the same diff-based precision
                # diff_connective/diff_lexical already use).
                diff_text = " ".join(candidate_diff_text)
                content_words = _content_words(diff_text, DISCOURSE_MARKER_EXTRAS)
                if content_words:
                    flags.append({
                        "span_id": s["span_id"], "type": "T", "text": s["text"],
                        "source_refs": [candidate_ref] if candidate_ref is not None else [],
                        "source_is_candidate_only": candidate_ref is not None,
                        "reason": (f"Nearly identical to its closest candidate source sentence "
                                   f"({candidate_overlap:.0%} content-word overlap) -- diverges by "
                                   f"{len(content_words)} content-bearing token(s): {diff_text!r}. "
                                   "Called metalinguistic, but this close to a real source sentence "
                                   "it may carry object-level content that belongs in R or C instead."),
                    })
            else:
                # No candidate strong enough to trust a diff against --
                # fall back to the original whole-sentence density check.
                content_words = _content_words(s["text"], DISCOURSE_MARKER_EXTRAS)
                if len(content_words) > 4:
                    flags.append({
                        "span_id": s["span_id"], "type": "T", "text": s["text"],
                        "source_refs": [candidate_ref] if candidate_ref is not None else [],
                        "source_is_candidate_only": candidate_ref is not None,
                        "reason": (f"{len(content_words)} content-bearing tokens -- unusually high for a "
                                   "metalinguistic span; may carry object-level content that belongs in R or C instead"),
                    })
    return flags


def run_injection_review(all_spans, borderline_flags, rate, decisions, source_sentences=None):
    """Interactive borderline-classification review: prints each flagged
    span and lets the researcher keep or reclassify it. Mutates all_spans
    entries in place. Every choice is appended to the decisions log via
    `decisions.record(...)` for later audit -- it does not change the
    interactive flow. Shared by phase2.ipynb and run_pipeline.py so
    there's one implementation, not two.

    `source_sentences`, if given (the second value classify_condensation
    returns), lets each flag also print the source sentence(s) it was
    matched against -- the researcher otherwise has to keep the whole
    source open in another window to judge whether a "keep or reassign"
    call is right."""
    if not borderline_flags:
        print("No borderline classifications flagged.")
        return

    print(f"\n{len(borderline_flags)} borderline classification(s) flagged for review:")

    spans_by_id = {s["span_id"]: s for s in all_spans}

    for flag in borderline_flags:

        print(f"\n  {flag['span_id']} ({flag['type']}): {flag['reason']}")
        print(f'    "{flag["text"]}"')
        if source_sentences is not None:
            label = "closest candidate (not an official match)" if flag.get("source_is_candidate_only") else "source"
            for idx in flag.get("source_refs", []):
                if 0 <= idx < len(source_sentences):
                    print(f'    {label}[{idx}]: "{source_sentences[idx]}"')

        new_type = input(
            "  Keep as classified, or reclassify? [ENTER = keep / F / T / R / C]: "
        ).strip().upper()

        decisions.record(
            step="injection_review", decision_type="borderline_reclassification",
            prompt="Keep as classified, or reclassify?",
            options=["", "F", "T", "R", "C"], choice=new_type,
            extra={
                "rate": rate, "span_id": flag["span_id"],
                "original_type": flag["type"],
                "new_type": new_type if new_type in ("F", "T", "R", "C") else flag["type"],
            },
        )

        if new_type in ("F", "T", "R", "C"):
            spans_by_id[flag["span_id"]]["type"] = new_type


# ============================================================
# CLUSTER COVERAGE
# ============================================================

def compute_injection_stats(all_spans, condensed_text, source_text):
    """Two independent sanity-check statistics, reported side by side
    (not one in place of the other -- they measure different things):
    non_injected_pct is the word share NOT covered by any classified F/T/R/C
    span; verbatim_overlap_pct is a mechanical token-alignment scan against
    the source, independent of the classification above."""
    injected_words = sum(len(s["text"].split()) for s in all_spans)
    condensed_words = count_words(condensed_text)
    non_injected_pct = round(100 * (1 - injected_words / condensed_words), 1) if condensed_words else 0.0

    verbatim_matched, verbatim_total, _ = scan_verbatim_overlap(condensed_text, source_text)
    verbatim_overlap_pct = round(100 * verbatim_matched / verbatim_total, 1) if verbatim_total else 0.0

    return {
        "non_injected_pct": non_injected_pct,
        "verbatim_overlap_pct": verbatim_overlap_pct,
        "verbatim_matched_tokens": verbatim_matched,
        "verbatim_total_tokens": verbatim_total,
    }


def _cluster_hit_counts(text, clusters, stemmer):
    """Token-occurrence hit counts per cluster, for one text -- how many
    times each of the cluster's stems appears as a token (via the same
    tokenize-and-stem approach used elsewhere in this pipeline), plus
    n-gram phrase-occurrence counts added into the same tally (n-grams
    aren't part of the original spec this mirrors, but this repo's
    clusters carry them alongside stems, so they're counted the same way
    the presence-only version already did -- just as counts, not booleans)."""
    tokens = re.findall(r"[A-Za-z']+", text)
    stem_counts = Counter(stemmer.stem(w.lower()) for w in tokens)
    text_lower = text.lower()

    hits = {}
    for cluster in clusters:
        key_terms = list(cluster.get("stems", [])) + list(cluster.get("ngrams", []))
        count = 0
        for term in key_terms:
            normalized = term.rstrip("*").lower()
            count += text_lower.count(normalized) if " " in normalized else stem_counts[normalized]
        hits[cluster["name"]] = count
    return hits


def compute_cluster_coverage(phase1_state, condensed_text, source_text, stemmer, tolerance_pp=5.0):
    """Per Phase 1 cluster: what share of all cluster-vocabulary token
    occurrences in the source belong to this cluster (source_pct -- the
    target, i.e. the relative emphasis the source itself gives this
    cluster), vs. the same share in the condensation (actual_pct).
    Mirrors the original PEEL spec's cluster_coverage(): occurrence-count
    shares, not presence/absence, so a cluster reduced to a single passing
    mention doesn't register the same as one actually preserved at its
    original weight. OK if actual is within +/-tolerance_pp of the
    source's own share; DARK if a cluster with real presence in the
    source (>0 hits) gets zero hits in the condensation; WARN otherwise."""
    clusters = phase1_state.get("clusterDefs", [])
    source_hits = _cluster_hit_counts(source_text, clusters, stemmer)
    condensed_hits = _cluster_hit_counts(condensed_text, clusters, stemmer)

    source_total = sum(source_hits.values())
    condensed_total = sum(condensed_hits.values())

    report = {}
    for cluster in clusters:
        name = cluster["name"]
        if not (cluster.get("stems") or cluster.get("ngrams")):
            continue

        source_pct = round(100 * source_hits[name] / source_total, 1) if source_total else 0.0
        actual_pct = round(100 * condensed_hits[name] / condensed_total, 1) if condensed_total else 0.0
        delta = round(actual_pct - source_pct, 1)

        if source_hits[name] > 0 and condensed_hits[name] == 0:
            status = "DARK"
        elif abs(delta) <= tolerance_pp:
            status = "OK"
        else:
            status = "WARN"

        report[name] = {
            "target": source_pct, "actual": actual_pct, "delta": delta, "status": status,
        }
    return report
