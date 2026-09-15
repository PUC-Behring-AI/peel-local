"""Source-vs-Summary comparison: builds a document-profile table plus,
per approved condensation rate, a word cloud, a keyword-in-context
concordance for the researcher-confirmed collocation pair(s), and a
collocates table -- positioned side by side against the same analyses
run on the source alone. Cluster coverage (does the summary preserve the
source's relative emphasis on each cluster?) reuses
phase2/condense.py's own compute_cluster_coverage()/
condensation_report.build_coverage_table rather than re-deriving the
same comparison a second time -- that function already answers exactly
this question for every approved rate.
"""

import html as html_lib

from phase2 import condense
from phase2.condensation_report import build_coverage_table
from phase2.pipeline import lexical_density
from phase3 import distant_reading as dr

esc = html_lib.escape


def build_document_profile_table(documents):
    """documents: list of (label, doc) spaCy Doc pairs, source first.
    Word count uses condense.count_words() (a plain whitespace split) --
    the same function Phase 2's own condensation report, run_pipeline.py,
    and pipeline_session.py already use for every other word count in
    this pipeline -- so this table's numbers agree with those instead of
    silently using a different, smaller, spaCy-alpha-token count."""
    rows = []
    for label, doc in documents:
        word_count = condense.count_words(doc.text)
        alpha_tokens = [t for t in doc if t.is_alpha]
        unique = len({t.text.lower() for t in alpha_tokens})
        density = round(lexical_density(doc), 3)
        rows.append(
            f'<tr><td style="padding:4px 10px 4px 0;">{esc(label)}</td>'
            f'<td style="padding:4px 10px 4px 0;text-align:right;">{word_count}</td>'
            f'<td style="padding:4px 10px 4px 0;text-align:right;">{unique}</td>'
            f'<td style="padding:4px 0;text-align:right;">{density}</td></tr>'
        )
    return f'''<table style="border-collapse:collapse;font-size:0.9em;">
<thead><tr><th style="text-align:left;padding:4px 10px 4px 0;">Document</th>
<th style="text-align:right;padding:4px 10px 4px 0;">Words</th>
<th style="text-align:right;padding:4px 10px 4px 0;">Unique words</th>
<th style="text-align:right;padding:4px 0;">Lexical density</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table>'''


def build_comparison_section(source_doc, rate_docs, phase1_state, source_text, stemmer,
                              stopwords, selected_pairs):
    """rate_docs: list of (rate, condensed_text, condensed_doc) for every
    approved rate, in order. Returns an HTML string with the whole
    'Source vs Summary' section."""
    documents = [("Source", source_doc)] + [(f"Summary at {rate}%", doc) for rate, _text, doc in rate_docs]

    parts = ["<h4>Document profile</h4>", build_document_profile_table(documents)]

    for rate, condensed_text, _doc in rate_docs:
        coverage = condense.compute_cluster_coverage(phase1_state, condensed_text, source_text, stemmer)
        parts.append(f"<h4>Cluster coverage &mdash; {rate}%</h4>")
        parts.append(build_coverage_table(coverage))

    # Word clouds (source vs. each summary) used to be rendered here, but
    # now appear earlier, in the Distant Reading section's own side-by-side
    # comparison (phase3/pipeline.py::build_phase3_report) -- not repeated.

    if selected_pairs:
        parts.append("<h4>Contexts &mdash; selected collocation pair(s)</h4>")
        for pair in selected_pairs:
            parts.append(
                f'<p style="font-size:0.9em;margin:1rem 0 0.3rem;">'
                f'<strong>{esc(pair["term_a"])}</strong> near <strong>{esc(pair["term_b"])}</strong></p>'
            )
            for label, doc in documents:
                parts.append(f'<p style="font-size:0.8em;color:#666;margin:6px 0 2px;">{esc(label)}</p>')
                parts.append(dr.build_contexts_table(doc, stemmer, pair["term_a"], pair["term_b"]))

        # Every term across every selected pair gets its own Collocates
        # table, not just the first pair's term_a -- deduplicated (the
        # same term can anchor more than one pair, e.g. "human" appearing
        # in several pairs) while preserving first-appearance order.
        anchors = []
        seen_anchors = set()
        for pair in selected_pairs:
            for term in (pair["term_a"], pair["term_b"]):
                if term not in seen_anchors:
                    seen_anchors.add(term)
                    anchors.append(term)

        parts.append("<h4>Collocates</h4>")
        parts.append(
            '<p style="font-size:0.85em;color:#666;margin:0 0 0.8rem;max-width:60em;">'
            "Each term below is one side of a pair you selected above -- every distinct term_a/term_b "
            "across your selected pair(s), not just the pairs themselves. For each one, every document "
            "(source and each summary) gets its own table: every occurrence of that term is scanned for "
            "any word within 5 tokens (a short fixed list of function words excluded), and the words found "
            "nearby are ranked by how often they occurred there. This runs independently per term and per "
            "document -- it is not restricted to the pairing you picked the term from, and it is not the "
            "same proximity/confound scan that produced the candidate list above.</p>"
        )
        for anchor in anchors:
            parts.append(f'<p style="font-size:0.9em;margin:1rem 0 0.3rem;"><strong>{esc(anchor)}</strong></p>')
            for label, doc in documents:
                parts.append(f'<p style="font-size:0.8em;color:#666;margin:6px 0 2px;">{esc(label)}</p>')
                parts.append(dr.build_collocates_table(doc, stemmer, anchor))

    return "\n".join(parts)
