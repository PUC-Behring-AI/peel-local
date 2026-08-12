"""Renders condense.py's condensation + classification results into an
HTML fragment, a standalone browser preview, a plain-text verification
report, and a markup-free plain summary.

CRITICAL -- FRAGMENT-FIRST: the primary output (`build_condensation_fragment`)
is an HTML *fragment*: no <!DOCTYPE>, <html>, <head>, <body>, <style> block,
or CSS class names. Every visual property is a style="..." attribute. This
is what lets common/standalone_report.py embed it as-is into its own
page without any style collisions.
"""

import html
import json
import re
from collections import Counter

esc = html.escape

PIPELINE_VERSION = "peel-local-phase2-condensation v1.0"

S = {
    # Typography
    "wrap": "font-family:Georgia,serif;max-width:820px;margin:0 auto;"
            "padding:0 0 4rem;line-height:1.85;color:#1c1a18;",
    "h1": "font-family:system-ui,sans-serif;font-size:1.3rem;font-weight:700;"
          "line-height:1.2em;margin:0 0 0.3rem;color:#1c1a18;",
    "authors": "font-family:system-ui,sans-serif;font-size:0.9rem;color:#666;"
               "margin:0 0 1.2rem;font-style:italic;",
    "h2": "font-family:system-ui,sans-serif;font-size:1.05rem;font-weight:700;"
          "color:#1a5c7a;margin:2.2rem 0 0.5rem;"
          "border-bottom:1px solid #d0e4ed;padding-bottom:0.2rem;",
    "h3": "font-family:system-ui,sans-serif;font-size:0.95rem;font-weight:600;"
          "color:#2a6080;margin:1.6rem 0 0.4rem;",
    "p": "font-family:Georgia,serif;margin:0 0 0.3rem 0;line-height:1.85;"
         "color:#1c1a18;",
    "defn": "font-family:Georgia,serif;background:#eef3f7;"
            "border-left:3px solid #1a5c7a;padding:0.7rem 1rem;"
            "margin:1rem 0;font-size:0.95rem;line-height:1.75;",

    # Metadata block
    "meta": "font-size:0.8rem;color:#777;margin:1.5rem 0 0.8rem;"
            "border-left:3px solid #ccc;padding-left:1rem;",
    "meta_td1": "padding:0.18rem 0.9rem 0.18rem 0;vertical-align:top;"
                "color:#aaa;white-space:nowrap;min-width:7rem;",
    "meta_td2": "padding:0.18rem 0.9rem 0.18rem 0;vertical-align:top;",

    # Legend
    "legend": "font-size:0.78rem;color:#555;margin:0.5rem 0 0;line-height:2.4;"
              "font-family:system-ui,sans-serif;",
    "sw": "display:inline-block;width:1rem;height:0.8rem;border-radius:2px;"
          "vertical-align:middle;margin-right:3px;font-size:0;line-height:0;",
    "hr": "border:none;border-top:1px solid #ddd;margin:0 0 1.5rem;",

    # Injection highlights -- four distinct, low-opacity, mnemonic hues:
    # green (F, low risk), blue (T, informational), gold (R, caution),
    # red (C, risk). Border style (none/dotted/dashed/solid) is a
    # redundant, colour-independent cue for the same F<T<R<C risk order.
    "inj_f": "background:rgba(42,125,58,0.18);",
    "inj_t": "background:rgba(70,130,180,0.20);border-bottom:1px dotted #2f5d80;",
    "inj_r": "background:rgba(212,160,23,0.22);border-bottom:1px dashed #8a6810;",
    "inj_c": "background:rgba(196,68,68,0.22);border-bottom:2px solid #8a2f2f;",

    # Inline source toggles -- collapsed-by-default provenance via native
    # <details>/<summary>. No <script>, no onclick.
    "c_details": "margin:0 0 1.2rem 0;font-size:0.85rem;",
    "c_summary": "cursor:pointer;color:#2a7d3a;font-family:system-ui,"
                 "sans-serif;font-size:0.7rem;font-weight:600;"
                 "text-transform:uppercase;letter-spacing:0.03em;",
    "c_inset": "background:#f0ede6;border-left:3px solid #2a7d3a;"
               "padding:0.5rem 0.8rem;margin:0.3rem 0 0;"
               "font-size:0.85rem;line-height:1.6;color:#444;"
               "font-family:Georgia,serif;",
    "r_details": "margin:0 0 1.2rem 0;font-size:0.85rem;",
    "r_summary": "cursor:pointer;color:#4a6ea8;font-family:system-ui,"
                 "sans-serif;font-size:0.7rem;font-weight:600;"
                 "text-transform:uppercase;letter-spacing:0.03em;",
    "r_inset": "background:#eef0f5;border-left:3px solid #4a6ea8;"
               "padding:0.5rem 0.8rem;margin:0.3rem 0 0;"
               "font-size:0.85rem;line-height:1.6;color:#444;"
               "font-family:Georgia,serif;",

    # Cluster report
    "cr_div": "margin-top:3rem;border-top:1px solid #ddd;padding-top:1rem;"
              "font-size:0.8rem;color:#888;font-family:system-ui,sans-serif;",
    "cr_th": "padding:0.25rem 0.6rem;border-bottom:1px solid #eee;"
             "text-align:left;color:#aaa;font-weight:normal;",
    "cr_td": "padding:0.25rem 0.6rem;border-bottom:1px solid #eee;text-align:left;",
    "cr_ok": "padding:0.25rem 0.6rem;border-bottom:1px solid #eee;"
             "text-align:left;color:#2a7d3a;",
    "cr_warn": "padding:0.25rem 0.6rem;border-bottom:1px solid #eee;"
               "text-align:left;color:#b07d2a;",
    "cr_dark": "padding:0.25rem 0.6rem;border-bottom:1px solid #eee;"
               "text-align:left;color:#8b3a2a;font-weight:bold;",

    "app_h": "font-size:0.85rem;color:#aaa;font-weight:normal;margin:0 0 0.5rem;",
}


# ============================================================
# STRUCTURAL BLOCK PARSER
# ============================================================
# No longer guesses at a title/author line from the condensed text's first
# two lines -- that positional heuristic ("first line = title, next =
# author") was fragile in practice: a condensation that doesn't open with
# a literal one-line title (the normal case -- build_condensation_prompt
# never asks the model for one) had its real opening paragraph(s)
# misdetected as "h1"/"authors" instead. The report's title/author/date
# now come from an explicit, separately-collected value (see
# run_source_metadata_setup/extract_source_metadata) -- every line here is
# ordinary body content.

_RE_H2 = re.compile(r"^\s*(\d)\.?\s*[\t ]+(.+)$")        # "1\tIntroduction" or "1. Introduction"
_RE_H3 = re.compile(r"^\s*(\d+\.\d+)\.?\s*[\t ]+(.+)$")  # "2.1\tTitle" or "2.1. Title"
_RE_DEFN = re.compile(r"^\s*Definition \d")


def parse_condensed_blocks(condensed_text):
    """Returns a list of (block_type, text) tuples.
    block_type in {'h2', 'h3', 'defn', 'p'}."""
    lines = condensed_text.split("\n")
    blocks = []
    buf = []

    def flush():
        if buf:
            blocks.append(("p", " ".join(buf).strip()))
            buf.clear()

    for line in lines:
        stripped = line.strip()
        if not stripped:
            flush()
            continue
        if _RE_DEFN.match(line):
            flush()
            blocks.append(("defn", stripped))
        elif _RE_H3.match(line):
            flush()
            blocks.append(("h3", stripped))
        elif _RE_H2.match(line):
            flush()
            blocks.append(("h2", stripped))
        else:
            buf.append(stripped)
    flush()
    return blocks


# ============================================================
# SPAN RENDERING
# ============================================================

def _spans_in_block(block_text, all_spans):
    return [s for s in all_spans if s["text"] in block_text]


def render_spans(block_text, all_spans):
    """Wraps each classified span's text, if found verbatim in block_text,
    with its S['inj_*'] style. A span not found in this block (e.g. the
    LLM's output line-wrapped a sentence differently than the block
    parser's line-joining) is left unhighlighted -- a disclosed limitation
    of substring-based span placement."""
    html_text = esc(block_text)
    for span in all_spans:
        escaped_span_text = esc(span["text"])
        if not escaped_span_text or escaped_span_text not in html_text:
            continue
        style = S[f"inj_{span['type'].lower()}"]
        wrapped = f'<span style="{style}">{escaped_span_text}</span>'
        html_text = html_text.replace(escaped_span_text, wrapped, 1)
    return html_text


def render_c_toggle(span_id, source_texts, max_shown=5):
    """Shows up to max_shown source sentences directly; any remainder is
    tucked behind a second, nested native <details> "Show N more" toggle
    (same no-<script> disclosure pattern as the outer one, just nested),
    since C spans can now trace to more than one contributing source."""
    shown, extra = source_texts[:max_shown], source_texts[max_shown:]
    items = "".join(f'<p style="margin:0 0 0.3rem">{esc(s)}</p>' for s in shown)

    more = ""
    if extra:
        extra_items = "".join(f'<p style="margin:0 0 0.3rem">{esc(s)}</p>' for s in extra)
        more = (
            f'<details style="{S["c_details"]}">'
            f'<summary style="{S["c_summary"]}">Show {len(extra)} more source(s)</summary>'
            f'<div style="{S["c_inset"]}">{extra_items}</div></details>'
        )

    return (
        f'<details style="{S["c_details"]}">'
        f'<summary style="{S["c_summary"]}">Show source (ᶜ{esc(span_id)})</summary>'
        f'<div style="{S["c_inset"]}">{items}{more}</div></details>'
    )


def render_r_toggle(span_id, source_text):
    return (
        f'<details style="{S["r_details"]}">'
        f'<summary style="{S["r_summary"]}">Show source (ʳ{esc(span_id)})</summary>'
        f'<div style="{S["r_inset"]}"><p style="margin:0">{esc(source_text)}</p></div></details>'
    )


_BLOCK_TAGS = {"p": ("p", "p"), "defn": ("div", "defn")}


def _render_blocks_with_toggles(body_blocks, all_spans, source_sentences):
    rendered = []
    for block_type, text in body_blocks:
        if block_type == "h2":
            rendered.append(f'<h2 style="{S["h2"]}">{esc(text)}</h2>')
            continue
        if block_type == "h3":
            rendered.append(f'<h3 style="{S["h3"]}">{esc(text)}</h3>')
            continue
        if block_type not in _BLOCK_TAGS:
            continue

        tag, style_key = _BLOCK_TAGS[block_type]
        rendered.append(f'<{tag} style="{S[style_key]}">{render_spans(text, all_spans)}</{tag}>')

        for span in _spans_in_block(text, all_spans):
            refs = [source_sentences[i] for i in span["source_refs"] if 0 <= i < len(source_sentences)]
            if span["type"] == "C" and refs:
                rendered.append(render_c_toggle(span["span_id"], refs))
            elif span["type"] == "R" and refs:
                rendered.append(render_r_toggle(span["span_id"], refs[0]))

    return "\n".join(rendered)


# ============================================================
# META / LEGEND / COVERAGE TABLE
# ============================================================

def build_meta_legend(corpus_name, source_word_count, rate, condensed_word_count,
                       phase1_json_name, span_counts):
    return f"""
<div style="{S['meta']}">
  <table style="border-collapse:collapse">
    <tbody>
    <tr><td style="{S['meta_td1']}">Source</td>
        <td style="{S['meta_td2']}">{esc(corpus_name)} &middot; {source_word_count} words</td></tr>
    <tr><td style="{S['meta_td1']}">Condensation</td>
        <td style="{S['meta_td2']}">{rate}% &middot; {condensed_word_count} words</td></tr>
    <tr><td style="{S['meta_td1']}">Phase&nbsp;1&nbsp;JSON</td>
        <td style="{S['meta_td2']}">{esc(phase1_json_name)}</td></tr>
    <tr><td style="{S['meta_td1']}">Injections</td>
        <td style="{S['meta_td2']}">F={span_counts.get('F', 0)} &middot; T={span_counts.get('T', 0)} &middot; R={span_counts.get('R', 0)} &middot; C={span_counts.get('C', 0)}</td></tr>
    <tr><td style="{S['meta_td1']}">Pipeline</td>
        <td style="{S['meta_td2']}">{esc(PIPELINE_VERSION)}</td></tr>
    </tbody>
  </table>
</div>
<div style="{S['legend']}">
  <span style="{S['sw']}background:rgba(42,125,58,0.85)">&nbsp;</span>
  <b>F &mdash; Framing</b>&nbsp;
  Whole sentence otherwise matches a source sentence; the only difference
  is &le;4 inserted/changed tokens, all purely connective (or, absent any
  source match, a freestanding connective fragment on its own).
  <em>Low epistemic risk.</em> &nbsp;&nbsp;&nbsp;
  <span style="{S['sw']}background:rgba(70,130,180,0.85);border-bottom:1px dotted #2f5d80">&nbsp;</span>
  <b>T &mdash; Transition</b>&nbsp;
  Metalinguistic sentence about the text's argument or structure, or a
  whole sentence that otherwise matches a source sentence with &le;6
  inserted/changed tokens that include genuine (non-connective) content.
  <em>Medium risk.</em> &nbsp;&nbsp;&nbsp;
  <span style="{S['sw']}background:rgba(212,160,23,0.85);border-bottom:1px dashed #8a6810">&nbsp;</span>
  <b>R &mdash; Reformulation</b>&nbsp;
  Single source sentence paraphrased. Click "Show source" directly below
  to reveal it.
  <em>Medium-high risk.</em> &nbsp;&nbsp;&nbsp;
  <span style="{S['sw']}background:rgba(196,68,68,0.85);border-bottom:2px solid #8a2f2f">&nbsp;</span>
  <b>C &mdash; Compression</b>&nbsp;
  Multiple source sentences collapsed. Click "Show source"
  directly below to reveal the passage it compresses &mdash; collapsed by
  default, one click to open, no JS.
  <em>High risk.</em> &nbsp;&nbsp;&nbsp;
  <span style="{S['sw']}background:transparent;border:1px solid #bbb">&nbsp;</span>
  <b>Unhighlighted text</b>&nbsp;
  Verbatim from the source, word for word -- the taxonomy only classifies
  non-verbatim spans, so nothing here has been added, reworded, or
  compressed.
  <em>Minimal risk.</em>
</div>
<p style="font-size:0.7rem;color:#999;font-family:system-ui,sans-serif;margin:0.3rem 0 0;">Classification is heuristic, not LLM-judged -- see the borderline flags and human report for disclosed uncertainty.</p>
<hr style="{S['hr']}">
"""


def _status_style(status):
    return {"OK": S["cr_ok"], "WARN": S["cr_warn"], "DARK": S["cr_dark"]}[status]


def build_coverage_table(coverage_report):
    rows = []
    for name, r in coverage_report.items():
        rows.append(
            f'<tr><td style="{S["cr_td"]}">{esc(name)}</td>'
            f'<td style="{S["cr_td"]}">{r["target"]}%</td>'
            f'<td style="{S["cr_td"]}">{r["actual"]}%</td>'
            f'<td style="{S["cr_td"]}">{r["delta"]:+.1f}pp</td>'
            f'<td style="{_status_style(r["status"])}">{esc(r["status"])}</td></tr>'
        )

    return f"""
<div style="{S['cr_div']}">
  <p style="{S['app_h']}"><b>Cluster coverage</b></p>
  <table style="border-collapse:collapse;width:100%">
    <tbody>
    <tr>
      <th style="{S['cr_th']}">Cluster</th>
      <th style="{S['cr_th']}">Target</th>
      <th style="{S['cr_th']}">Actual</th>
      <th style="{S['cr_th']}">Delta</th>
      <th style="{S['cr_th']}">Status</th>
    </tr>
    {''.join(rows)}
    </tbody>
  </table>
</div>
"""


# ============================================================
# FINAL ASSEMBLY
# ============================================================

def build_condensation_fragment(condensed_text, all_spans, source_sentences, coverage_report,
                                 corpus_name, rate, source_word_count, phase1_json_name,
                                 title=None, authors=None, date=None):
    """title/authors/date: the source document's own metadata, collected
    explicitly (typed in, or LLM-extracted -- see
    run_source_metadata_setup/extract_source_metadata), not guessed from
    the condensed text. Optional so existing callers (e.g. phase2.ipynb)
    keep working unmodified, falling back to corpus_name with no byline."""
    blocks = parse_condensed_blocks(condensed_text)

    section_blocks = _render_blocks_with_toggles(blocks, all_spans, source_sentences)
    span_counts = Counter(s["type"] for s in all_spans)
    meta_legend = build_meta_legend(
        corpus_name, source_word_count, rate, len(condensed_text.split()),
        phase1_json_name, span_counts,
    )
    cov_table = build_coverage_table(coverage_report)

    title_text = title or corpus_name
    byline_text = " &middot; ".join(esc(v) for v in (authors, date) if v and v != "Unclear")

    return f"""<div style="{S['wrap']}">
{meta_legend}
<h1 style="{S['h1']}">{esc(title_text)}</h1>
<p style="{S['authors']}">{byline_text}</p>
{section_blocks}
{cov_table}
</div>"""


def build_standalone_preview(fragment_html, corpus_name):
    return f"""<!doctype html>
<html>
<head><meta charset="utf-8"><title>{esc(corpus_name)} -- Condensation Preview</title></head>
<body style="margin:2rem auto;max-width:900px;background:#fdfcfa;">
{fragment_html}
</body>
</html>"""


def build_human_report(all_spans, borderline_flags, coverage_report,
                        verbatim_overlap_pct, non_injected_pct, sanity_issues=None):
    span_counts = Counter(s["type"] for s in all_spans)
    lines = [
        "PEEL-Local Phase 2 Condensation -- Verification Report",
        "=" * 56,
        "",
        f"Injection spans: F={span_counts.get('F', 0)}  T={span_counts.get('T', 0)}  "
        f"R={span_counts.get('R', 0)}  C={span_counts.get('C', 0)}",
        f"Non-injected (word share outside classified spans): {non_injected_pct}%",
        f"Independent verbatim-overlap scan: {verbatim_overlap_pct}%",
        "",
        "Generation sanity checks (invented vocabulary, runaway tokens, repetition loops):",
    ]
    if not sanity_issues:
        lines.append("  none")
    else:
        for issue in sanity_issues:
            lines.append(f"  - {issue}")

    lines += ["", "Borderline classification flags:"]
    if not borderline_flags:
        lines.append("  none")
    else:
        for f in borderline_flags:
            lines.append(f"  {f['span_id']} ({f['type']}): {f['reason']}")
            lines.append(f'    "{f["text"]}"')

    lines += ["", "Cluster coverage:"]
    for name, r in coverage_report.items():
        lines.append(f"  {name}: target {r['target']}%, actual {r['actual']}% ({r['delta']:+.1f}pp) -- {r['status']}")

    lines += ["", "Compression (C) and Reformulation (R) spans -- source citations:"]
    for s in all_spans:
        if s["type"] in ("C", "R"):
            lines.append(f"  {s['span_id']} ({s['type']}): {s['text']}")
            lines.append(f"    {s['justification']}")

    return "\n".join(lines) + "\n"


def build_plain_summary(blocks):
    """Title/authors/body text, no injection markup -- built from the
    already-parsed blocks, never by stripping tags from the fragment."""
    lines = []
    for block_type, text in blocks:
        lines.append(text)
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def save_condensation_outputs(paths_dict, fragment, preview, report, summary, injection_report_data):
    with open(paths_dict["html_fragment"], "w", encoding="utf-8") as f:
        f.write(fragment)
    with open(paths_dict["html_preview"], "w", encoding="utf-8") as f:
        f.write(preview)
    with open(paths_dict["human_report"], "w", encoding="utf-8") as f:
        f.write(report)
    with open(paths_dict["plain_summary"], "w", encoding="utf-8") as f:
        f.write(summary)
    with open(paths_dict["injection_report"], "w", encoding="utf-8") as f:
        json.dump(injection_report_data, f, indent=2, ensure_ascii=False)
