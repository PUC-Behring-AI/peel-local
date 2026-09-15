"""Single-corpus 'Distant Reading' analyses -- Python-native
reimplementations of the questions a retired set of Voyant tool cells
used to answer (Reader, Cirrus, Trends, Phrases, CorpusTerms, Contexts,
Collocates), adapted from a researcher-supplied specification for a
separate, Voyant-based PEEL system this repo doesn't use. Every builder
here works on a single document (the source, or one condensation
summary), so phase3/comparison.py reuses them unchanged, once per
document, for the source-vs-summary comparison.

Two deliberate simplifications from the original spec, both a direct
consequence of not being bound to Voyant's own tool set any more:
CollocatesGraph's node-link graph is replaced by a plain co-occurrence
table (the researcher's own real-Voyant testing of the source spec found
the table form worked better for this exact use case anyway), and
Bubblelines is not reproduced as a separate chart -- it answers the same
"how does a cluster's presence vary across the corpus" question as
Trends, just as bubbles instead of a line.
"""

import base64
import html as html_lib
import math
from collections import Counter
from io import BytesIO

from scipy.stats import kurtosis, skew
from wordcloud import WordCloud

from phase2.condense import FUNCTION_WORD_SET, _cluster_hit_counts
from phase3 import terms as pterms

esc = html_lib.escape


# ============================================================
# READER -- full text with cluster-term highlighting
# ============================================================

def _mark_background(colors) -> str:
    """CSS background for a highlighted token: the plain translucent
    single color when only one cluster claims it (unchanged from before
    multi-cluster support), or a hard-edged striped gradient combining
    every claiming cluster's color when more than one does -- same 55
    (~33%) alpha suffix per stripe as the single-color case, so a shared
    stem/n-gram visibly shows all its clusters instead of one silently
    winning and the rest vanishing (see build_reader_html)."""
    if len(colors) == 1:
        return f"background:{colors[0]}55;"
    n = len(colors)
    stops = []
    for i, color in enumerate(colors):
        start = round(100 * i / n, 3)
        end = round(100 * (i + 1) / n, 3)
        stops.append(f"{color}55 {start}%")
        stops.append(f"{color}55 {end}%")
    return f"background:linear-gradient(90deg, {', '.join(stops)});"


def build_reader_html(doc, clusterdefs, colors_by_cluster, stemmer, max_chars=20000):
    """Full text with cluster-significant stems/n-grams highlighted and
    colour-coded by cluster -- token-level stem/lemma matching (see
    terms.py), not a regex approximation. Capped at max_chars for report
    size; states the truncation explicitly rather than silently cutting
    the text.

    A stem/n-gram claimed by more than one cluster (a real possibility --
    see recluster_noise's per-occurrence embedding, which doesn't
    guarantee the clean one-stem-one-cluster partition the primary
    clustering pass does) is highlighted with EVERY claiming cluster's
    color, striped together (_mark_background), with the tooltip listing
    every cluster name -- not silently attributed to whichever cluster
    happens to come first alphabetically, which is what a plain
    first-write-wins dict would do."""
    stems, lemmas = pterms.token_forms(doc, stemmer)

    stem_to_cluster = {}
    ngram_token_colors = {}
    for cluster in clusterdefs:
        color = colors_by_cluster.get(cluster["name"], "#888")
        name = cluster["name"]
        for stem in cluster.get("stems", []):
            normalized = stem.rstrip("*").lower()
            stem_to_cluster.setdefault(normalized, {})[name] = color
        for gram in cluster.get("ngrams", []):
            for span in pterms.term_match_spans(stems, lemmas, gram):
                for k in span:
                    ngram_token_colors.setdefault(k, {})[name] = color

    tokens = list(doc)
    pieces = []
    truncated = False
    char_budget = max_chars
    for idx, token in enumerate(tokens):
        if char_budget <= 0:
            truncated = True
            break
        char_budget -= len(token.text_with_ws)

        color_map = ngram_token_colors.get(idx)
        if not color_map and stems[idx]:
            color_map = stem_to_cluster.get(stems[idx])

        if color_map:
            colors = list(color_map.values())
            names = ", ".join(color_map.keys())
            pieces.append(
                f'<mark style="{_mark_background(colors)}padding:0 1px;border-radius:2px;" '
                f'title="{esc(names)}">{esc(token.text)}</mark>{esc(token.whitespace_)}'
            )
        else:
            pieces.append(esc(token.text) + esc(token.whitespace_))

    body = f'<div style="white-space:pre-wrap;line-height:1.9;font-family:Georgia,serif;">{"".join(pieces)}</div>'
    note = f'<p><em>Text truncated at ~{max_chars:,} characters for report size.</em></p>' if truncated else ""
    return note + body


# ============================================================
# LAYOUT -- side-by-side comparison (source vs. condensation(s))
# ============================================================

def build_side_by_side_html(items, min_width="260px"):
    """items: [(label, inner_html), ...]. Lays them out in the same
    wrapping flex row comparison.py's per-rate word clouds already used,
    one column per item with its label above. A single item is returned
    unwrapped -- so a report with no approved condensation yet renders
    exactly as it did before this existed, not inside a pointless
    one-column flex container."""
    if len(items) <= 1:
        return items[0][1] if items else ""
    cols = "".join(
        f'<div style="flex:1 1 {min_width};"><p style="font-size:0.85em;color:#666;margin:0 0 4px;">{esc(label)}</p>{html}</div>'
        for label, html in items
    )
    return f'<div style="display:flex;flex-wrap:wrap;gap:1rem;">{cols}</div>'


# ============================================================
# CIRRUS -- word cloud
# ============================================================

def build_wordcloud_html(text, stopwords, width=700, height=350, max_words=100):
    if not text.strip():
        return "<p><em>No text to render.</em></p>"
    wc = WordCloud(
        width=width, height=height, background_color="white",
        stopwords=stopwords, max_words=max_words, collocations=False,
    )
    try:
        wc.generate(text)
    except ValueError:
        return "<p><em>Not enough distinct words after stopword filtering.</em></p>"

    buf = BytesIO()
    wc.to_image().save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return (
        f'<img src="data:image/png;base64,{b64}" alt="word cloud" '
        'style="max-width:100%;border:1px solid #ddd;border-radius:4px;">'
    )


# ============================================================
# TRENDS -- cluster prevalence across the corpus, binned
# ============================================================

def bin_cluster_frequencies(text, clusterdefs, stemmer, n_bins=5):
    """Splits text into n_bins equal-length word segments and counts
    each cluster's stem/n-gram hits per segment, reusing
    phase2/condense.py's own _cluster_hit_counts so this agrees with how
    the rest of the pipeline already counts cluster-term occurrences."""
    words = text.split()
    if not words:
        return {c["name"]: [0] * n_bins for c in clusterdefs}
    bin_size = max(1, math.ceil(len(words) / n_bins))
    raw_bins = [" ".join(words[i:i + bin_size]) for i in range(0, len(words), bin_size)]
    raw_bins = (raw_bins + [""] * n_bins)[:n_bins]

    freqs = {c["name"]: [] for c in clusterdefs}
    for bin_text in raw_bins:
        hits = _cluster_hit_counts(bin_text, clusterdefs, stemmer)
        for name, count in hits.items():
            freqs[name].append(count)
    return freqs


def build_trend_chart_svg(bin_freqs, colors_by_cluster, width=640, height=220, pad=40):
    n_bins = len(next(iter(bin_freqs.values()), []))
    if n_bins == 0 or not any(any(v) for v in bin_freqs.values()):
        return "<p><em>Not enough text to chart.</em></p>"
    max_val = max((v for values in bin_freqs.values() for v in values), default=0) or 1

    plot_w, plot_h = width - 2 * pad, height - 2 * pad

    def px(i):
        return pad + (i / max(n_bins - 1, 1)) * plot_w

    def py(v):
        return pad + plot_h - (v / max_val) * plot_h

    lines = []
    for name, values in bin_freqs.items():
        color = colors_by_cluster.get(name, "#888")
        points = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(values))
        lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
        # A 2px line is a thin, hard-to-hit hover target -- an invisible
        # wide duplicate on top (same points) gives a real hit area while
        # staying visually identical, and carries the tooltip itself so
        # hovering anywhere along the line (not just on a point marker)
        # names the cluster it belongs to.
        lines.append(
            f'<polyline points="{points}" fill="none" stroke="transparent" stroke-width="12">'
            f'<title>{esc(name)}</title></polyline>'
        )
        for i, v in enumerate(values):
            lines.append(
                f'<circle cx="{px(i):.1f}" cy="{py(v):.1f}" r="3" fill="{color}">'
                f'<title>{esc(name)}: {v}</title></circle>'
            )

    axis = (
        f'<line x1="{pad}" y1="{pad + plot_h}" x2="{pad + plot_w}" y2="{pad + plot_h}" stroke="#ccc"/>'
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{pad + plot_h}" stroke="#ccc"/>'
    )
    labels = "".join(
        f'<text x="{px(i):.1f}" y="{pad + plot_h + 16}" font-size="10" text-anchor="middle" fill="#888">seg {i + 1}</text>'
        for i in range(n_bins)
    )

    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" style="max-width:{width}px;" role="img">'
        f'{axis}{"".join(lines)}{labels}</svg>'
    )


# ============================================================
# PHRASES -- n-gram frequency table
# ============================================================

def _is_subsequence(short_gram, long_gram):
    n = len(short_gram)
    return any(long_gram[i:i + n] == short_gram for i in range(len(long_gram) - n + 1))


def build_phrase_table(doc, stopwords, min_n=2, max_n=5, top_n=25):
    tokens = [t.text.lower() for t in doc if t.is_alpha and t.text.lower() not in stopwords]

    counter = Counter()
    for n in range(min_n, max_n + 1):
        for i in range(len(tokens) - n + 1):
            counter[tuple(tokens[i:i + n])] += 1

    candidates = [(g, c) for g, c in counter.items() if c >= 2]
    by_length = sorted(candidates, key=lambda gc: (len(gc[0]), gc[1]), reverse=True)

    kept = []
    for gram, count in by_length:
        if any(len(k) > len(gram) and _is_subsequence(gram, k) for k, _ in kept):
            continue
        kept.append((gram, count))
    kept.sort(key=lambda gc: gc[1], reverse=True)
    kept = kept[:top_n]

    if not kept:
        return "<p><em>No recurring phrases found.</em></p>"

    rows = "".join(
        f'<tr><td style="padding:4px 10px 4px 0;">{esc(" ".join(g))}</td>'
        f'<td style="padding:4px 0;text-align:right;">{c}</td></tr>'
        for g, c in kept
    )
    return f'''<table style="border-collapse:collapse;font-size:0.88em;">
<thead><tr><th style="text-align:left;padding:4px 10px 4px 0;">Phrase</th><th style="text-align:right;">Count</th></tr></thead>
<tbody>{rows}</tbody></table>'''


# ============================================================
# CORPUSTERMS -- frequency, relative frequency, distribution shape
# ============================================================

def _sparkline_svg(values, width=90, height=18, color="#4E79A7"):
    if not values or max(values) == 0:
        return f'<svg width="{width}" height="{height}"></svg>'
    max_v = max(values)
    bar_w = width / len(values)
    bars = "".join(
        f'<rect x="{i * bar_w:.1f}" y="{height - (v / max_v) * height:.1f}" '
        f'width="{max(bar_w - 1, 1):.1f}" height="{max((v / max_v) * height, 1):.1f}" fill="{color}"/>'
        for i, v in enumerate(values)
    )
    return f'<svg width="{width}" height="{height}">{bars}</svg>'


def build_term_stats_table(doc, clusterdefs, stemmer, n_bins=8, top_n=25):
    """Per significant term: raw/relative frequency, plus peakedness
    (kurtosis) and skewness of its per-segment frequency vector -- how
    concentrated vs. evenly spread, and front-loaded vs. back-loaded, the
    same two questions the retired CorpusTerms/Document Terms tool's
    columns answered.

    Excludes any term with raw_freq <= 1, not just 0 -- a single
    occurrence always lands in exactly one of the n_bins segments, which
    is a degenerate (not just uninformative) distribution: its
    peakedness/skewness come out identical for every freq-1 term
    regardless of which segment that lone occurrence happened to land in,
    so the shape columns carry no real signal there. This also keeps a
    much shorter document (e.g. a heavy condensation) from having its
    table dominated by a long tail of one-off mentions competing for
    top_n slots against terms with an actual distribution to describe."""
    stems, lemmas = pterms.token_forms(doc, stemmer)
    total_tokens = len(stems)
    bin_size = max(1, math.ceil(total_tokens / n_bins)) if total_tokens else 1

    terms = sorted({term for c in clusterdefs for term in list(c.get("stems", [])) + list(c.get("ngrams", []))})

    rows = []
    for term in terms:
        positions = pterms.term_positions(stems, lemmas, term)
        raw_freq = len(positions)
        if raw_freq <= 1:
            continue

        per_bin = [0] * n_bins
        for p in positions:
            b = min(p // bin_size, n_bins - 1)
            per_bin[b] += 1

        rel_freq = round(100 * raw_freq / total_tokens, 4) if total_tokens else 0.0
        if len(set(per_bin)) > 1:
            peakedness = round(float(kurtosis(per_bin, fisher=True)), 2)
            skewness = round(float(skew(per_bin)), 2)
        else:
            peakedness = skewness = 0.0

        rows.append((term, raw_freq, rel_freq, peakedness, skewness, per_bin))

    rows.sort(key=lambda r: r[1], reverse=True)
    rows = rows[:top_n]

    if not rows:
        return "<p><em>No significant terms found in this document.</em></p>"

    body = "".join(
        f'<tr><td style="padding:4px 10px 4px 0;">{esc(term)}</td>'
        f'<td style="padding:4px 10px 4px 0;text-align:right;">{raw_freq}</td>'
        f'<td style="padding:4px 10px 4px 0;text-align:right;">{rel_freq}%</td>'
        f'<td style="padding:4px 10px 4px 0;text-align:right;">{peakedness}</td>'
        f'<td style="padding:4px 10px 4px 0;text-align:right;">{skewness}</td>'
        f'<td style="padding:4px 0;">{_sparkline_svg(per_bin)}</td></tr>'
        for term, raw_freq, rel_freq, peakedness, skewness, per_bin in rows
    )
    return f'''<table style="border-collapse:collapse;font-size:0.85em;">
<thead><tr><th style="text-align:left;padding:4px 10px 4px 0;">Term</th>
<th style="text-align:right;padding:4px 10px 4px 0;">Raw freq.</th>
<th style="text-align:right;padding:4px 10px 4px 0;">Rel. freq.</th>
<th style="text-align:right;padding:4px 10px 4px 0;" title="Kurtosis of the term's per-segment frequency -- how concentrated vs. evenly spread it is">Peakedness</th>
<th style="text-align:right;padding:4px 10px 4px 0;" title="Skew of the term's per-segment frequency -- front-loaded vs. back-loaded">Skewness</th>
<th style="text-align:left;padding:4px 0;">Distribution</th></tr></thead>
<tbody>{body}</tbody></table>'''


# ============================================================
# CONTEXTS -- keyword-in-context concordance
# ============================================================

def build_contexts_table(doc, stemmer, term_a, term_b=None, window=8, proximity_n=5, max_rows=25):
    """KWIC concordance for term_a, optionally restricted to occurrences
    within proximity_n tokens of term_b -- a genuine collocation context,
    not just every occurrence of term_a alone."""
    stems, lemmas = pterms.token_forms(doc, stemmer)
    positions_a = pterms.term_positions(stems, lemmas, term_a)
    if term_b:
        positions_b = pterms.term_positions(stems, lemmas, term_b)
        positions_a = pterms.positions_near(positions_a, positions_b, proximity_n)

    tokens = list(doc)
    rows = []
    for i in positions_a[:max_rows]:
        left = "".join(t.text_with_ws for t in tokens[max(0, i - window):i])
        term_text = tokens[i].text
        right = "".join(t.text_with_ws for t in tokens[i + 1:i + 1 + window])
        rows.append((left, term_text, right))

    if not rows:
        return "<p><em>No occurrences found.</em></p>"

    body = "".join(
        f'<tr><td style="text-align:right;padding:3px 6px;color:#555;">&hellip;{esc(l.strip())}</td>'
        f'<td style="padding:3px 6px;font-weight:bold;color:#1a5c7a;white-space:nowrap;">{esc(t)}</td>'
        f'<td style="padding:3px 6px;color:#555;">{esc(r.strip())}&hellip;</td></tr>'
        for l, t, r in rows
    )
    return f'<table style="border-collapse:collapse;font-size:0.85em;width:100%;"><tbody>{body}</tbody></table>'


# ============================================================
# COLLOCATES -- co-occurrence table (replaces CollocatesGraph)
# ============================================================

def build_collocates_table(doc, stemmer, anchor_term, window=5, top_n=20):
    stems, lemmas = pterms.token_forms(doc, stemmer)
    positions = pterms.term_positions(stems, lemmas, anchor_term)
    if not positions:
        return "<p><em>No occurrences found.</em></p>"

    tokens = list(doc)
    counter = Counter()
    for i in positions:
        for j in range(max(0, i - window), min(len(tokens), i + window + 1)):
            if j == i or not tokens[j].is_alpha:
                continue
            w = tokens[j].text.lower()
            if w in FUNCTION_WORD_SET:
                continue
            counter[w] += 1

    # The anchor's own occurrence count is the same for every row (it's a
    # property of the anchor in this document, not of the co-occurring
    # word) -- state it once above the table instead of repeating it on
    # every row.
    occurrence_note = (
        f'<p style="font-size:0.85em;color:#888;margin:0 0 4px;">'
        f'{len(positions)} occurrence{"s" if len(positions) != 1 else ""} of &ldquo;{esc(anchor_term)}&rdquo;</p>'
    )

    rows = counter.most_common(top_n)
    if not rows:
        return occurrence_note + "<p><em>No notable co-occurring terms.</em></p>"

    body = "".join(
        f'<tr><td style="padding:3px 8px 3px 0;">{esc(w)}</td>'
        f'<td style="padding:3px 0;text-align:right;">{c}</td></tr>'
        for w, c in rows
    )
    return f'''{occurrence_note}<table style="border-collapse:collapse;font-size:0.85em;">
<thead><tr><th style="text-align:left;padding:3px 8px 3px 0;">Co-occurring term</th>
<th style="text-align:right;padding:3px 0;">Count</th></tr></thead>
<tbody>{body}</tbody></table>'''
