"""Converts a corpus input file to plain text, regardless of source
format, before it's ever placed at data/<corpus>/raw/<corpus>.txt.

Every entry point that places a corpus into that raw/ folder -- the web
interface's upload handler (webapp/pipeline_session.py) and
run_pipeline.py's --input -- funnels through convert_to_text() here, so
there's one implementation of "what counts as a supported input format"
and how each one becomes text, not several.

.txt is a plain passthrough decode. .md/.markdown gets a light,
regex-based markdown-to-prose pass (not a full CommonMark parser --
same "good enough for this pipeline's needs" approach
phase0/clean_corpus.py already takes for PDF-extraction artifacts):
strips syntax markers, keeps the underlying words, since the analysis
downstream (stemming, embeddings, GlossBERT) wants readable prose, not
markup. .pdf is extracted page by page via pypdf -- this commonly
introduces exactly the running-header/footer/page-number/hyphenation
artifacts phase0/clean_corpus.py already knows how to clean, so running
Phase 0 cleaning on a PDF-sourced corpus is worth doing even more than
usual.
"""

import io
import re

SUPPORTED_EXTENSIONS = (".txt", ".md", ".markdown", ".pdf")


def convert_to_text(raw_bytes: bytes, filename: str) -> str:
    """raw_bytes: the file's raw content. filename: the original
    filename (or anything ending in the right extension) -- used only to
    pick the conversion path. Returns extracted plain text.

    Raises ValueError for an unsupported extension, or if PDF text
    extraction finds no extractable text at all (e.g. a scanned/
    image-only PDF -- this module does no OCR). Raises
    UnicodeDecodeError, uncaught, for a .txt/.md file that isn't valid
    UTF-8 -- callers already handle that distinctly from other failures."""
    ext = _extension(filename)
    if ext == ".txt":
        return _normalize_newlines(raw_bytes.decode("utf-8"))
    if ext in (".md", ".markdown"):
        return _markdown_to_text(_normalize_newlines(raw_bytes.decode("utf-8")))
    if ext == ".pdf":
        return _normalize_newlines(_pdf_to_text(raw_bytes))
    raise ValueError(
        f"Unsupported file type {ext or '(no extension)'!r} -- expected one of: "
        + ", ".join(SUPPORTED_EXTENSIONS)
    )


def _extension(filename: str) -> str:
    name = filename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    if "." not in name:
        return ""
    return "." + name.rsplit(".", 1)[-1].lower()


def _normalize_newlines(text: str) -> str:
    """Reading raw_bytes always happens in binary mode (a PDF needs that,
    and one code path has to serve all three formats) -- which, unlike a
    text-mode read, does no universal-newline translation. A
    Windows-authored \\r\\n file would otherwise carry literal \\r
    characters through the whole pipeline, then get *doubled* the next
    time the text is written in Python's text mode (which translates
    every \\n to os.linesep, turning an existing \\r\\n into \\r\\r\\n).
    Normalizing once, right after decoding, avoids both."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _pdf_to_text(raw_bytes: bytes) -> str:
    from pypdf import PdfReader  # imported lazily -- only a PDF input needs it

    reader = PdfReader(io.BytesIO(raw_bytes))
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n\n".join(p.strip() for p in pages if p.strip())
    if not text.strip():
        raise ValueError(
            "No extractable text found in this PDF -- it may be a scanned/image-only "
            "document, which this converter can't OCR."
        )
    return text


# ---------------------------------------------------------------------
# MARKDOWN -> PLAIN TEXT
# ---------------------------------------------------------------------

def _markdown_to_text(text: str) -> str:
    # Fenced code blocks: drop the ``` / ~~~ fence marker lines (whatever
    # follows on that same line, e.g. a language tag), keep the content
    # between them -- safer than discarding it outright.
    text = re.sub(r"^(```|~~~)[^\n]*\n", "", text, flags=re.MULTILINE)

    # Images: ![alt](url "title") -> alt
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)
    # Links: [text](url "title") -> text
    text = re.sub(r"\[([^\]]*)\]\(([^)]*)\)", r"\1", text)
    # Reference-style link/image definitions on their own line: [ref]: url "title"
    text = re.sub(r"^[ \t]*\[[^\]]+\]:\s*\S+.*$", "", text, flags=re.MULTILINE)
    # Reference-style usage: [text][ref] -> text
    text = re.sub(r"\[([^\]]*)\]\[[^\]]*\]", r"\1", text)

    # Inline HTML tags
    text = re.sub(r"<[^>]+>", "", text)

    # Headings: leading #'s
    text = re.sub(r"^#{1,6}[ \t]*", "", text, flags=re.MULTILINE)
    # Blockquote markers
    text = re.sub(r"^>[ \t]?", "", text, flags=re.MULTILINE)
    # List markers (bullet or ordered)
    text = re.sub(r"^[ \t]*[-*+][ \t]+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[ \t]*\d+[.)][ \t]+", "", text, flags=re.MULTILINE)
    # Horizontal rules (a line of 3+ -, *, or _, optionally spaced out)
    text = re.sub(r"^[ \t]*([-*_])[ \t]*(\1[ \t]*){2,}$", "", text, flags=re.MULTILINE)

    # Emphasis/strong/strikethrough -- strip markers, keep the text.
    # Longest markers first so **bold** isn't left half-stripped by the
    # single-marker pass.
    text = re.sub(r"(\*\*\*|___)(.+?)\1", r"\2", text)
    text = re.sub(r"(\*\*|__)(.+?)\1", r"\2", text)
    text = re.sub(r"(?<!\w)(\*|_)(?!\1)(.+?)\1(?!\w)", r"\2", text)
    text = re.sub(r"~~(.+?)~~", r"\1", text)
    # Inline code
    text = re.sub(r"`([^`]*)`", r"\1", text)

    # Table separator rows: |---|---|, :---:|---, etc.
    text = re.sub(r"^[ \t]*\|?[ \t:|-]*-[ \t:|-]*\|?[ \t]*$", "", text, flags=re.MULTILINE)
    # Remaining table pipes: drop a leading/trailing one, turn the rest
    # into spaces -- keeps every cell's words as readable text even
    # though the column structure itself is lost.
    text = re.sub(r"^[ \t]*\|", "", text, flags=re.MULTILINE)
    text = re.sub(r"\|[ \t]*$", "", text, flags=re.MULTILINE)
    text = text.replace("|", " ")

    # Collapse the blank lines left behind by stripped fence/HR/table-separator lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"
