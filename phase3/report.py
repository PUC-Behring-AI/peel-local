"""Assembles Phase 3's standalone HTML report: cluster/colour legend,
single-corpus Distant Reading analyses, and (if any condensation rates
were approved) the Source-vs-Summary comparison section. Same
self-contained, inline-style, cleanly-styled-page convention as
common/standalone_report.py -- openable offline, no external service.
"""

import datetime
import html as html_lib

from phase1.pipeline import hex_to_rgb

esc = html_lib.escape

_PAGE_STYLE = """
<style>
  :root { color-scheme: light dark; }
  body {
    font-family: Georgia, serif;
    max-width: 980px;
    margin: 2.5rem auto;
    padding: 0 1.5rem 4rem;
    line-height: 1.7;
    color: #1c1a18;
    background: #fdfcfa;
  }
  header.report-header {
    border-bottom: 2px solid #1a5c7a;
    margin-bottom: 1.5rem;
    padding-bottom: 0.8rem;
  }
  header.report-header h1 {
    font-family: system-ui, sans-serif;
    font-size: 1.6rem;
    margin: 0 0 0.2rem;
    color: #1a5c7a;
  }
  header.report-header p {
    font-family: system-ui, sans-serif;
    color: #666;
    font-size: 0.9rem;
    margin: 0;
  }
  section.report-section {
    margin: 2.2rem 0;
  }
  section.report-section > h2 {
    font-family: system-ui, sans-serif;
    font-size: 1.15rem;
    color: #1a5c7a;
    border-bottom: 1px solid #d0e4ed;
    padding-bottom: 0.3rem;
  }
  section.report-section h4 {
    font-family: system-ui, sans-serif;
    font-size: 0.95rem;
    color: #2a6080;
    margin: 1.4rem 0 0.5rem;
  }
  @media (prefers-color-scheme: dark) {
    body { background: #16181c; color: #e6e2d8; }
    header.report-header { border-bottom-color: #5aa9d6; }
    header.report-header h1, section.report-section > h2, section.report-section h4 { color: #7cc0ec; }
    header.report-header p { color: #9aa0a6; }
    section.report-section > h2 { border-bottom-color: #2a3138; }
  }
</style>
"""


def build_cluster_legend_html(clusterdefs, colors_by_cluster):
    rows = []
    for cluster in clusterdefs:
        color = colors_by_cluster.get(cluster["name"], "#888")
        r, g, b = hex_to_rgb(color)
        stems = cluster.get("stems", [])
        shown = ", ".join(esc(s) for s in stems[:8])
        more = "&hellip;" if len(stems) > 8 else ""
        rows.append(
            '<tr><td style="padding:4px 10px 4px 0;">'
            f'<span style="display:inline-block;width:12px;height:12px;border-radius:2px;'
            f'background:rgb({r},{g},{b});"></span></td>'
            f'<td style="padding:4px 10px 4px 0;font-weight:bold;color:rgb({r},{g},{b});">{esc(cluster["name"])}</td>'
            f'<td style="padding:4px 0;font-size:0.85em;color:#666;">{shown}{more}</td></tr>'
        )
    return f'<table style="border-collapse:collapse;"><tbody>{"".join(rows)}</tbody></table>'


def build_cross_cluster_html(clusterdefs):
    stem_to_clusters = {}
    for cluster in clusterdefs:
        for stem in cluster.get("stems", []):
            stem_to_clusters.setdefault(stem, []).append(cluster["name"])
    cross = {s: names for s, names in stem_to_clusters.items() if len(names) > 1}
    if not cross:
        return "<p><em>No stem appears in more than one cluster.</em></p>"
    items = "".join(
        f'<li><code>{esc(s)}</code> &rarr; {", ".join(esc(n) for n in names)}</li>'
        for s, names in sorted(cross.items())
    )
    return f"<ul>{items}</ul>"


def build_provenance_html(stopword_count, auto_authors, collocation_candidates, selected_pairs, skip_reason=None):
    lines = [f"<p>{stopword_count} stopword(s) in effect for word clouds/frequency tables.</p>"]
    if auto_authors:
        lines.append(
            "<p>Automatically included as author-name stopwords (PERSON entities mentioned "
            f"3+ times, not researcher-confirmed): {', '.join(esc(n) for n in auto_authors)}.</p>"
        )
    else:
        lines.append("<p>No candidate author name cleared the automatic mention threshold.</p>")

    if collocation_candidates:
        lines.append(
            f"<p>{len(collocation_candidates)} real source collocation(s) found among cluster-significant terms.</p>"
        )
    if selected_pairs:
        pairs_str = "; ".join(f'{esc(p["term_a"])} &harr; {esc(p["term_b"])}' for p in selected_pairs)
        lines.append(f"<p>Selected for the Contexts/Collocates comparison: {pairs_str}.</p>")
    elif skip_reason:
        lines.append(
            f'<p><strong>Contexts/Collocates comparison skipped:</strong> {esc(skip_reason)}.</p>'
        )
    else:
        lines.append("<p>No collocation pair was selected (no condensation was generated to compare against).</p>")
    return "\n".join(lines)


def build_distant_reading_report(corpus_name, clusterdefs, colors_by_cluster,
                                  reader_html, wordcloud_html, trend_chart_html,
                                  phrase_table_html, term_stats_html,
                                  comparison_html, provenance_html):
    generated_at = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    comparison_section = ""
    if comparison_html:
        comparison_section = f'''
<section class="report-section">
  <h2>Source vs. Summary</h2>
  {comparison_html}
</section>'''

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{esc(corpus_name)} -- Distant Reading Report</title>
{_PAGE_STYLE}
</head>
<body>
<header class="report-header">
  <h1>{esc(corpus_name)} &mdash; Distant Reading Report</h1>
  <p>Phase 3 &middot; generated {generated_at}</p>
</header>

<section class="report-section">
  <h2>Semantic clusters</h2>
  {build_cluster_legend_html(clusterdefs, colors_by_cluster)}
  <h4>Cross-cluster stems</h4>
  {build_cross_cluster_html(clusterdefs)}
</section>

<section class="report-section">
  <h2>Distant reading</h2>
  <h4>Word cloud</h4>
  {wordcloud_html}
  <h4>Cluster prevalence across the corpus</h4>
  {trend_chart_html}
  <h4>Recurring phrases</h4>
  {phrase_table_html}
  <h4>Term frequency &amp; distribution shape</h4>
  {term_stats_html}
  <h4>Full text (highlighted)</h4>
  <details>
    <summary style="cursor:pointer;font-weight:600;color:#1a5c7a;">Show full text</summary>
    <div style="margin-top:0.8rem;">
      {reader_html}
    </div>
  </details>
</section>
{comparison_section}
<section class="report-section">
  <h2>Provenance</h2>
  {provenance_html}
</section>

</body>
</html>"""


def save_distant_reading_report(html, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
