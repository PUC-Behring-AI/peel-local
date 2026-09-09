# AI prompts & model-input catalog

Every place PEEL-Local hands text to an AI model, and what it does with the
response. Kept separate from [FUNCTIONS.md](../FUNCTIONS.md) (which
catalogs *every* function) because this is specifically an audit trail for
reproducibility: given a corpus and a model, a reader should be able to
find the exact text a model saw at each step without reading the source.

Scope: **prompts** (text sent to a generative model expecting a text
response) and **classification inputs** (structured input sent to a model
expecting a score/label, e.g. GlossBERT). Embedding-only calls
(`SentenceTransformer.encode`) aren't prompts in this sense -- no
instructions, no generation -- so they're listed separately in their own
section for completeness rather than mixed in above.

Every prompt/input below is built by a dedicated `build_*`/`glossbert_predict*`
function and passed to either `common/ollama_client.py::generate()` (LLM
text generation) or a GlossBERT forward pass -- never inlined ad hoc at the
call site, so there is exactly one place per prompt to read or change it.

---

## Phase 1 -- Semantic analysis

### GlossBERT word-sense disambiguation (per-occurrence, default)

| | |
|---|---|
| **Function** | `glossbert_predict(occurrence, tokenizer, model, device, pos_map, max_synsets)` |
| **File** | [`phase1/pipeline.py:111`](../phase1/pipeline.py) |
| **Called from** | `run_glossbert_analysis` (same file), once per sampled occurrence |
| **Model** | GlossBERT (`jvomiranda/GlossBERT_Checkpoint`, Hugging Face; default -- user-configurable via `--glossbert-model`/`GLOSSBERT_MODEL_ID`/webapp field) |
| **Workflow step** | Phase 1, "GlossBERT WSD inference" -- runs once GlossBERT is loaded and stem occurrences are mapped to sentences, before flagged-term review |

**Input** (one `(sentence, gloss)` pair per candidate WordNet sense, up to
`max_synsets` forward passes per occurrence, all sharing one windowed
sentence -- see below):

- *Sentence A* -- a **word-window** around the target word, not
  necessarily the whole sentence (see "Windowing" below), bracket-marked
  via a case-insensitive regex match and wrapped in quotes
- *Sentence B* -- the candidate WordNet synset's `.definition()` text
- Tokenized together with `truncation=True, max_length=128, padding="max_length"`

**Output**: `softmax(logits)[0][1]` per candidate (probability the gloss
matches the marked word's sense in context); candidates are ranked by this
score, and the top one is compared against WordNet's own default
(first-listed) sense to decide whether the word gets auto-accepted or
flagged for review.

**Windowing (fixes real truncation cases).** Originally, *Sentence A* was
the occurrence's whole sentence, verbatim. HuggingFace's default
pair-truncation strategy trims from the end of whichever of the two
sequences is longer, which for a long sentence means the *tail* gets cut
-- regardless of where the target word actually sits. Verified on real
Boisseau data: in one 114-word sentence, the words `"explain"` and
`"able"` (occurring late in the sentence) were silently dropped from
GlossBERT's input entirely -- not truncated-but-marked, just *absent*,
with no quote-marking anywhere in what the model saw -- while `"hand"`
and `"mobilising"` (occurring earlier in the same sentence) survived
intact. GlossBERT was disambiguating the dropped words with zero local
context.

Fix (`_glossbert_words_budget` + `_window_around_match`,
[`phase1/pipeline.py:82-108`](../phase1/pipeline.py)): instead of the
whole sentence, feed a **word-based window centered on the target word**.
The window size is computed once per occurrence from the *longest*
candidate gloss among the synsets being scored (so every candidate sense
is judged against the identical sentence context -- only the gloss
differs) via:

```
gloss_tokens = tokenizer(worst_case_gloss) token count, minus [CLS]/[SEP]
budget = 128 - 3 (specials) - 2 (quote-mark tokens) - gloss_tokens
words_each_side = budget / 1.6 (measured p95 tokens-per-word) / 2
```

For a sentence already short enough to fit whole, the requested window
exceeds the sentence's own length, so this is a byte-identical no-op --
behavior only changes for the sentences that were silently losing the
word before. Re-verified on the same real Boisseau sentence: all four
words (including the two previously dropped) now survive, using 68-88 of
the 128-token budget.

**Word-boundary marking (fixes marking the wrong word entirely).** The
regex that locates the target word within its sentence (to center the
window and insert the quote-marks) was an unanchored substring search --
`re.search(re.escape(word), sentence, re.IGNORECASE)`. For a short stem
whose letters occur inside an unrelated, more common word earlier in the
same sentence (e.g. `"ai"` inside `"main"`, `"again"`, `"certain"`,
`"maintain"`), `re.search` returns that *first* substring match, not the
real target -- windowing and quote-marking then happen around the wrong
word, and the real occurrence is never marked at all. Verified on real
Boisseau data: for the stem `"ai"`, one of five sampled occurrences was
the sentence *"...distinguishes three **main** forms of opacity...
machine learning-based AI systems."* -- the unanchored regex matched the
`"ai"` inside `"main"` (position 26) and completely ignored the real
`"AI"` later in the same sentence (position 120), producing the marked
text `"...three m "ai" n forms..."`, entirely unrelated to the actual
disambiguation target. A corpus-wide scan of this run's accepted terms
found 6 real occurrences (stems `"ai"` and `"al"`) affected the same way.

Fix: anchor the regex to word boundaries --
`re.search(r"\b" + re.escape(word) + r"\b", sentence, re.IGNORECASE)`, in
both `glossbert_predict` and `glossbert_predict_merged`. Re-verified on
the same real sentence: the fixed regex correctly matches the real `"AI"`
at position 120, ignoring the `"main"` substring entirely. This is a
context-correctness fix, not a confidence fix -- for the `"ai"` case
specifically, GlossBERT's own candidate scores remained closely
clustered (0.149-0.171 across 4 candidates) even with clean context,
because WordNet's sense inventory for `"ai"` is a poor fit for the
"artificial intelligence" acronym sense regardless of how good the
surrounding context is. Separately, `"al"` (from citation fragments like
"et al.") and similar non-content tokens can still reach WSD at all
because spaCy's `is_alpha` filter doesn't exclude citation abbreviations
-- a real, but different and still-open, source of poor-looking glosses
that this fix does not address.

**Pre-existing-quote stripping (fixes a possible second, indistinguishable
quote pair).** GlossBERT's paper marks the disambiguation target by
wrapping it in `"..."` within the sentence -- this is the model's *only*
signal for which word is being disambiguated. If the surrounding context
already contains a straight-quote pair of its own (a direct quotation,
scare-quotes, code-like text), the marked sentence would contain two
visually identical `"..."` spans with nothing to tell the model which one
is the real target. Checked directly against this corpus: Boisseau
contains zero literal straight double-quotes (`"`, U+0022) -- it uses
typographic curly quotes exclusively (`'`/`'`, `"`/`"`), which the
tokenizer maps to entirely different token ids than the marking character
(verified: `"` is token 1000; `"`/`"` are 1523/1524; `'`/`'` are
1520/1521 -- no collision), so this was not causing any of the poor
glosses observed in this run. It's a real robustness gap for other
corpora/converters that do preserve straight quotes, though.

Fix (`_window_around_match`, shared by both `glossbert_predict` and
`glossbert_predict_merged`): any literal `"` in the window text
(before/after the match, not the matched word itself) is replaced with a
space before the caller wraps the target in its own fresh pair --
`sentence[:match_start].replace('"', " ")` / same for the after-half.
Replaced with a space rather than deleted outright, so two words
separated only by a quote don't get fused together. Verified this is a
byte-identical no-op on real Boisseau data (0 straight quotes present --
e.g. `expert.n.01` still scores exactly 0.2779 on the same sentence used
to verify the windowing fix above) and correctly resolves a synthetic
collision case: `'She said "the river was calm" as they walked past the
bank yesterday.'` marking `"bank"` now produces `'She said the river was
calm as they walked past the "bank" yesterday.'` -- exactly one quote
pair, the real one -- instead of two indistinguishable pairs.

### GlossBERT word-sense disambiguation (merged, opt-in)

| | |
|---|---|
| **Function** | `glossbert_predict_merged(word, occurrences, tokenizer, model, device, pos_map, max_synsets)` |
| **File** | [`phase1/pipeline.py:161`](../phase1/pipeline.py) |
| **Called from** | `run_glossbert_analysis`, once per distinct word (instead of once per occurrence) when `merge_duplicate_word_occurrences=True` |
| **Model** | Same GlossBERT checkpoint as above |
| **Workflow step** | Phase 1, "GlossBERT WSD inference" -- opt-in alternative to the per-occurrence path, toggled at Phase 1 setup (`--merge-duplicate-word-occurrences` / `MERGE_DUPLICATE_WORD_OCCURRENCES` / webapp checkbox) |

**Input**: same `(text, gloss)` pairing as above, but tokenized with
`max_length=512` (`_GLOSSBERT_MERGED_MAX_LENGTH`) instead of 128, and
*Sentence A* is every one of the word's sampled occurrence-sentences,
each independently **windowed and marked** around its own instance of the
word (same `_window_around_match` as the per-occurrence path above, but
the shared 512-token budget is split N ways across all N occurrences
being merged), then joined with a space into one paragraph.

**Why 512, not 128.** GlossBERT is BERT-base underneath
(`max_position_embeddings=512`, confirmed directly from the checkpoint's
config), so 512 is the model's actual architectural ceiling -- not an
arbitrary increase. The per-occurrence path above deliberately stays at
128 (one sentence window per forward pass rarely needs more, and it runs
once per occurrence rather than once per merged group), but a merged
paragraph concatenates several occurrences' sentences and has
correspondingly more text to fit; giving it only the same 128-token
budget as a single occurrence left little room per occurrence once split
N ways.

Verified on a real 5-occurrence case ("expertise" in Boisseau, spread
across the title/intro/three later paragraphs): at the old 128-token
budget, only the first 2 of 5 marked occurrences survived truncation
(94/128 tokens went to a single sentence's tail). At 512 tokens, all 5 of
5 survive comfortably, using 216 of 512 tokens (30 words of window each
side per occurrence in this case, versus 6 at the old budget). Merging
still has a real ceiling -- splitting one shared budget more ways gives
each occurrence a smaller window -- but 512 pushes that ceiling far
enough out that it no longer bites for realistic occurrence counts.

---

## Phase 2 -- Condensation & verification

### Condensation generation prompt

| | |
|---|---|
| **Function** | `build_condensation_prompt(ordered_sentences, cluster_key_terms, target_words, corpus_name, source_text=None)` |
| **File** | [`phase2/condense.py:128`](../phase2/condense.py) |
| **Called from** | `attempt_condensation_trials`, once per generation trial (up to `max_trials` times per rate, plus again on escalation with `source_text` supplied) |
| **Model** | User-selected Ollama model (`--ollama-model` / `OLLAMA_MODEL` / webapp field; no fixed default -- whatever's installed locally) |
| **Workflow step** | Phase 2, "Generation trials & sanity checks" (and its escalated retry) |

**Template** (verbatim, `{...}` = interpolated):

```
You are condensing an excerpt from the corpus "{corpus_name}" to
approximately {target_words} words (+/-5%).

Below are the corpus's most informative sentences, in their original order,
and the key terms of each semantic cluster identified in prior analysis.
Write a single, coherent, readable condensation that:
1. Follows the argumentative structure implied by these sentences, not
necessarily their literal order.
2. Faithfully represents the source's argument, structure, and conclusions.
3. Reaches approximately {target_words} words.
4. Uses the source's own vocabulary and key terms wherever possible,
especially the cluster terms listed below, but is not constrained to do so.
5. Does not introduce examples, framings, or conclusions absent from the
material below.

Informative sentences (in corpus order):
{sentences_block}

Cluster key terms to try to preserve:
{terms_block}
[if escalated: For additional context, here is the full source text. Use it to
faithfully represent the argument, but keep prioritizing the informative
sentences and key terms above when deciding what to include:

{source_text}
]

Return only the condensed text, with no preamble or commentary.
```

`sentences_block` is `- {sentence}` per informative sentence;
`terms_block` is `- {cluster_name}: {top 15 terms}` per cluster.
`estimate_num_ctx` (`condense.py:112`) sizes the Ollama context window from
this prompt's length; it doesn't add to the prompt text itself.

### Adversarial review prompt

| | |
|---|---|
| **Function** | `build_adversarial_review_prompt(condensed_text, ordered_sentences, target_words, corpus_name)` |
| **File** | [`phase2/condense.py:263`](../phase2/condense.py) |
| **Called from** | `run_adversarial_review`, once per rate, after a trial is accepted and before it's saved |
| **Model** | User-selected Ollama model, independent from the generation model (`--adversarial-model` / `ADVERSARIAL_MODEL` / webapp field; defaults to the generation model if left unset) |
| **Workflow step** | Phase 2, "Adversarial review & save" -- the last thing that happens to a condensation's text before it's persisted and used by every downstream step |

**Template**:

```
You are auditing a condensation of the corpus "{corpus_name}", which was
generated to target approximately {target_words} words.

Below is the pool of informative sentences the condensation was supposed
to be built from, followed by the condensation itself. Check the
condensation for three specific defects:
1. Does it end abruptly, mid-sentence or mid-thought?
2. Does it contain garbled, repeated, or nonsensical sequences of words
or tokens?
3. Does its content drift substantially outside the topics/claims covered
by the informative sentences below (i.e. does it introduce ideas not
grounded in that pool)?

Informative sentences:
{sentences_block}

Condensation to audit:
{condensed_text}

If none of these three defects are present, respond with exactly:
VERDICT: OK

If any defect is present, fix ONLY that defect (make the smallest
possible edit; do not rewrite for style), keep the result as close as
possible to {target_words} words, and respond in exactly this format:
VERDICT: FIXED
ISSUES: <comma-separated short description of each defect found>
TEXT:
<the corrected condensation, and nothing else after it>
```

Response is parsed by `parse_adversarial_response` (`condense.py`, right
below the prompt builder); a response that doesn't parse, or whose fixed
text is empty/implausibly short, is treated as `VERDICT: OK` (original
text kept) -- see `run_adversarial_review`'s degeneracy guard.

### Source metadata extraction prompt

| | |
|---|---|
| **Function** | `extract_source_metadata(source_text, model, embedder_name="all-MiniLM-L6-v2", top_k=5, host=, lead_chunks=3)` |
| **File** | [`phase2/condense.py:532`](../phase2/condense.py) |
| **Called from** | `run_source_metadata_setup`, when the researcher opts to let the LLM determine title/author(s)/date rather than typing them in |
| **Model** | Same Ollama model chosen for condensation generation (metadata extraction has no separate model choice); retrieval uses `all-MiniLM-L6-v2` (Sentence-BERT, fixed, not user-configurable) |
| **Workflow step** | Phase 2, "Condensation & source setup" |

**Template**:

```
Below are excerpts from a document, retrieved because they are most likely
to contain its title, author(s), and publication date.

{retrieved}

Based only on these excerpts, respond in exactly this format (one line each,
nothing else):
TITLE: <the document's title, or Unclear>
AUTHOR(S): <the author(s)/byline, or Unclear>
DATE: <the publication date, or Unclear>
If you are not reasonably confident about a field, write Unclear for it -- do not guess.
```

`{retrieved}` is not the whole source: paragraph-level chunks are embedded
alongside one fixed synthetic query ("The document's title, author(s) or
byline, and publication date."), the `top_k=5` chunks by cosine similarity
are kept, and the first `lead_chunks=3` chunks are *always* included too
regardless of score (front matter is positionally reliable; on a
real-corpus test, pure semantic retrieval alone ranked the actual
title/author chunks 159th/124th of 181 -- see the function's docstring).
Retrieved chunks are joined with `\n\n---\n\n` before being interpolated.

---

## Non-prompt AI model usage (embeddings)

Not prompts (no instructions, no generated text) -- included for
completeness since they're still "AI models" in the pipeline. Sentence-BERT
never sees a bare stem string on its own for meaning -- every call below
embeds a constructed text that mixes the stem (as a label only), the real
inflected surface words that were observed for it (e.g. "cats", not "cat"),
and real sentence context, so the vector positions driving clustering are
shaped by actual usage, not the abstract stem token.

**Context-sentence capping (fixes real truncation cases, adapts to any
source).** All three embedding calls below feed `all-MiniLM-L6-v2`, whose
actual `max_seq_length` is **256** (measured directly, not the
commonly-assumed 128). Each embeds a variant of the same template, and
the "Contexts"/"Sentence" portion was originally an uncapped, verbatim
source sentence -- on real Boisseau data, this alone pushed 2 of the 3
longest stem-texts over 256 tokens, and because "Candidate senses" comes
*last* in the template, it's always what silently gets cut first: the
stem `system` (287 tokens) lost its entire "Candidate senses" section;
`observ` (271 tokens) kept only a few words of its first definition.

Fix: `_fit_contexts_to_budget(tokenizer, max_seq_length, fixed_parts,
contexts)` ([`phase1/pipeline.py:696-741`](../phase1/pipeline.py))
replaces a fixed, corpus-derived word-count cap with one computed **from
the actual text at call time**, so it fits the source in front of it
rather than one particular corpus's measured statistics:

1. Tokenize every OTHER piece of the template for real (label, observed
   words, candidate senses, punctuation) and sum their actual token
   counts -- `fixed_tokens`.
2. `budget = max_seq_length - 2 (special tokens) - fixed_tokens` is
   whatever's left over for context sentences. A stem whose candidate
   senses run long (more/longer definitions) automatically leaves less
   budget for contexts, and vice versa -- there's no fixed reservation
   per section, so nothing needs re-tuning if a different source's
   sections tend to run longer or shorter.
3. That budget is split across however many context sentences there are,
   and each sentence is capped using **its own** real tokens-per-word
   ratio (`tokenizer(sentence)` token count / word count), not a
   corpus-wide assumed ratio. A sentence with short common words keeps
   more words for the same token budget than one with long
   compound/technical terms or heavy subword-splitting -- this is what
   makes the cap adapt automatically across fields/languages instead of
   inheriting Boisseau-specific numbers. Demonstrated directly: under an
   identical artificial budget, a plain-English sentence (1.06
   tokens/word measured) kept 35 words while a dense technical-jargon
   sentence (3.21 tokens/word measured) kept only 12 -- same budget,
   different real ratio, different cap.

Applied at all three call sites below. Re-verified on the same two real
Boisseau stems that were previously broken: `system` now 247 tokens (was
287) with its definition intact; `observ` now 255 tokens (was 271), both
definitions intact -- both inside the 256 cap. If a stem's label +
observed words + candidate senses alone ever exceed the budget on their
own (pathological edge case), a `WARNING` prints and contexts fall back
to a 3-word-per-sentence minimum rather than failing silently.

**Silencing a harmless-but-alarming tokenizer warning.** `_fit_contexts_to_budget`
measures each RAW context sentence's real length before deciding how much
to trim it -- that measurement call originally had no `truncation=True`,
so a genuinely long real sentence (verified on Boisseau: one real
sentence is 295 tokens on its own) made `transformers` print "Token
indices sequence length is longer than the specified maximum sequence
length for this model (295 > 256). Running this sequence through the
model will result in indexing errors" on every such measurement. This
sounds like a crash risk but isn't one: it's printed by the raw
measurement call, not by the actual `embedder.encode()` call the trimmed
text is later fed to -- verified directly that (a) `SentenceTransformer.encode()`
never actually sees anything over `max_seq_length` here (0 of 131 real
Boisseau stem-embedding texts, and 0 of the real `recluster_noise` texts,
exceed 256 tokens with this run's actual config), and (b) even in a
synthetic worst case where it did, `encode()` silently truncates
internally rather than erroring (cosine similarity 0.98 between a 400-word
text and its 200-word prefix -- not NaN, not a crash). Fixed by adding
`truncation=True, max_length=max_seq_length` to the measurement call
itself: since the measured value is only ever compared against a much
smaller per-sentence budget share, capping it at `max_seq_length` changes
no trimming decision -- purely cosmetic, silences the warning with no
behavior change. The analogous (but so far only theoretical -- WordNet
glosses are always short) measurement in `_glossbert_words_budget` got
the same defensive fix for consistency.

### Stem embeddings for initial clustering (one point per stem)

| | |
|---|---|
| **Function** | `build_stem_embeddings(accepted_definitions, embedder_name)` |
| **File** | [`phase1/pipeline.py:710`](../phase1/pipeline.py) |
| **Called from** | Phase 1, "Save senses, embed & initial cluster" -- right after flagged-term review, before the first HDBSCAN pass |
| **Model** | Sentence-BERT (`SENTENCE_EMBEDDER`, default `all-MiniLM-L6-v2`, user-configurable via `--sentence-embedder`/`SENTENCE_EMBEDDER`/webapp field) |

Builds **one text per unique stem** (not one per instance -- a high-instance
stem would otherwise out-vote its own density requirement in the HDBSCAN
step that follows), from that stem's *accepted* instances only
(`accepted_definitions[stem]["instances"]` -- i.e. only words the
researcher already reviewed/confirmed at flagged-term review, or that
auto-agreed with WordNet's default sense):

```
observed_words = first 5 unique instance["word"] values (real surface forms, e.g. "cats")
contexts       = first 3 unique instance["sentence"] values (real sentences), each
                 capped by _fit_contexts_to_budget to whatever this stem's actual
                 label/observed-words/candidate-senses tokens leave in the 256-token budget
definitions    = first 5 unique instance["definition"] values (accepted WordNet glosses)

text = f"Stem: {stem}. Observed words: {', '.join(observed_words)}. "
       f"Contexts: {' '.join(contexts)}. Candidate senses: {' '.join(definitions)}"
```

`embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)`
produces one L2-normalized vector per stem, fed straight to
`cluster_stem_embeddings`'s HDBSCAN pass.

### Re-embedding for oversized-cluster resplitting (one point per stem, richer/fresh context)

| | |
|---|---|
| **Function** | `_split_oversized_clusters_once(clusters, stem_occurrences, embedder, tokenizer, model, device, ...)` |
| **File** | [`phase1/pipeline.py:823`](../phase1/pipeline.py) |
| **Called from** | `recluster_large_clusters`, Phase 1 "Recursive cluster refinement" |
| **Model** | Same Sentence-BERT model |

Same template (`f"Stem: {stem}. Observed words: ... Contexts: ... Candidate
senses: ..."`, contexts capped the same way), but two differences from the
initial embedding: (1) source data is **`stem_occurrences`** (every real
occurrence of the stem in the corpus), not `accepted_definitions` -- so a
stem's context here isn't limited to what the researcher happened to
review; (2) `candidate_senses` comes from **fresh `glossbert_predict()`
calls made right here**, taking the top-3 candidate definitions per
occurrence (not the single previously-accepted one) -- a deliberately
richer semantic signal meant to help HDBSCAN's leaf-mode pass actually
separate an oversized cluster into coherent pieces. Note
`glossbert_predict` here also benefits from the windowing fix described
above -- richer context no longer comes at the cost of the WSD calls that
produce these candidate senses losing their own target word to
truncation.

### Re-embedding noise stems (one point PER OCCURRENCE, not per stem)

| | |
|---|---|
| **Function** | `recluster_noise(noise_stems, stem_occurrences, embedder, tokenizer, model, device, ...)` |
| **File** | [`phase1/pipeline.py:970`](../phase1/pipeline.py) |
| **Called from** | Phase 1, "Recursive cluster refinement" (stems HDBSCAN's first pass discarded as noise) |
| **Model** | Same Sentence-BERT model |

Structurally different from the two above: embeds **one point per
occurrence**, not one aggregated point per stem --

```
for occurrence in stem_occurrences[stem]:
    results = glossbert_predict(occurrence, ...)
    sentence = _fit_contexts_to_budget(tokenizer, max_seq_length,
                   [prefix, suffix], [occurrence["sentence"]])[0]
    text = f"Word: {occurrence['word']}. Sentence: {sentence}. " \
           f"Candidate senses: {' '.join(c['definition'] for c in results[:3])}"
```

so a stem with 5 occurrences contributes 5 separate embedding points (all
labeled with that stem name) to the noise re-clustering pass, rather than
being collapsed to one. This gives HDBSCAN more points to find a genuine
small cluster among what was previously all discarded as noise. Only one
context sentence per text here (not three), but it was still uncapped in
length before this fix, so the same risk applied whenever that single
sentence ran long -- now it goes through the same adaptive budget-fitting
as the two stem-aggregated templates above.

### Source-metadata retrieval query

| | |
|---|---|
| **Function** | `extract_source_metadata(source_text, model, embedder_name="all-MiniLM-L6-v2", top_k=5, ...)` |
| **File** | [`phase2/condense.py:557,559,560`](../phase2/condense.py) |
| **Called from** | Phase 2, "Condensation & source setup" |
| **Model** | `all-MiniLM-L6-v2` (fixed, not user-configurable here) |

Embeds the source's paragraph-level chunks plus one fixed synthetic query
("The document's title, author(s) or byline, and publication date.") for
cosine-similarity retrieval -- see the Phase 2 section above for the full
retrieval-then-prompt mechanics.

---

## Quick reference

| Prompt/input | Phase / step | Function | Model choice |
|---|---|---|---|
| GlossBERT WSD (per-occurrence) | Phase 1 -- GlossBERT WSD inference | `glossbert_predict` | Fixed per run (`--glossbert-model`) |
| GlossBERT WSD (merged, opt-in) | Phase 1 -- GlossBERT WSD inference | `glossbert_predict_merged` | Same as above |
| Condensation generation | Phase 2 -- Generation trials & sanity checks | `build_condensation_prompt` | `--ollama-model` |
| Adversarial review | Phase 2 -- Adversarial review & save | `build_adversarial_review_prompt` | `--adversarial-model` (defaults to `--ollama-model`) |
| Source metadata extraction | Phase 2 -- Condensation & source setup | `extract_source_metadata` (builds prompt inline) | Same as `--ollama-model` |

Regenerate the pipeline diagrams (`docgraph/generate_all.py`) after
changing any of these -- `PEEL-Local-pipeline-full.html`'s node detail
panels quote each prompt's exact rule/formula and should stay in sync with
this file.
