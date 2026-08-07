"""Builds a self-contained, non-Voyant HTML report of a corpus's Phase 1 +
Phase 2 results -- the same breadth of information as the Voyant notebook
export (`common/voyant_notebook.py`), but as its own cleanly styled page,
not bound by Voyant/Spyral's inline-style-only, no-<style>-block
constraint. Generated alongside (not instead of) the Voyant notebook.
"""

import datetime
import html as html_lib

from common.voyant_notebook import build_cluster_results_html, build_optional_material_html

esc = html_lib.escape

_PAGE_STYLE = """
<style>
  :root { color-scheme: light dark; }
  body {
    font-family: Georgia, serif;
    max-width: 900px;
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
  @media (prefers-color-scheme: dark) {
    body { background: #16181c; color: #e6e2d8; }
    header.report-header { border-bottom-color: #5aa9d6; }
    header.report-header h1, section.report-section > h2 { color: #7cc0ec; }
    header.report-header p { color: #9aa0a6; }
    section.report-section > h2 { border-bottom-color: #2a3138; }
  }
</style>
"""


def build_standalone_report(phase1_state, fragment, corpus_name, rate, paths) -> str:
    """Returns a complete, self-contained HTML page combining the
    condensation fragment, Phase 1's cluster results, and a run summary --
    the non-Voyant equivalent of `voyant_notebook.build_voyant_notebook`.
    Reuses voyant_notebook's cluster/summary renderers rather than
    duplicating them; the condensation fragment itself is already valid
    inline-style HTML (from condensation_report.build_condensation_fragment)
    so it's embedded as-is."""
    generated_at = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    cluster_html = build_cluster_results_html(phase1_state)
    optional_html = build_optional_material_html(phase1_state, corpus_name, [rate], paths)

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{esc(corpus_name)} -- PEEL-Local Report ({rate}%)</title>
{_PAGE_STYLE}
</head>
<body>
<header class="report-header">
  <h1>{esc(corpus_name)} -- Condensation Report</h1>
  <p>{rate}% condensation &middot; generated {generated_at}</p>
</header>

<section class="report-section">
  <h2>Condensation</h2>
  {fragment}
</section>

<section class="report-section">
  <h2>Phase 1 semantic clusters</h2>
  {cluster_html}
</section>

<section class="report-section">
  <h2>Run summary</h2>
  {optional_html}
</section>

</body>
</html>"""


def save_standalone_report(html: str, path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
