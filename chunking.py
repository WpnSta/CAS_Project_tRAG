"""
Document loading, text cleaning, and chunking logic for the FDV RAG system.
Imported by all notebooks.
"""

import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration constants (imported by notebooks alongside the functions)
# ---------------------------------------------------------------------------

CHUNK_MIN_CHARS     = 150   # stubs smaller than this are merged upward
CHUNK_MAX_CHARS     = 800   # sections larger than this are split at paragraphs
CHUNK_OVERLAP_CHARS = 100   # overlap carried into the next chunk

# Lookahead split: keeps the heading at the start of each part
SECTION_HEADING_RE = re.compile(r"(?=^\d+(?:\.\d+)*\t)", re.MULTILINE)
_HEADING_PARSE_RE  = re.compile(r"^(\d+(?:\.\d+)*)\t([^\t\n]+)", re.MULTILINE)
_FILENAME_RE       = re.compile(r"^(\d+)[fi]?_(.+)$")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_filename(path):
    """
    Return (regulation_number, title) from a filename like '14_Bremsen.txt',
    '14f_Freins.txt', or '14i_Freni.txt'.
    """
    stem = Path(path).stem
    m = _FILENAME_RE.match(stem)
    if not m:
        return (None, stem)
    reg_num = f"R 300.{int(m.group(1))}"
    title   = m.group(2).replace("_", " ")
    return reg_num, title


def load_documents(data_dir, language):
    """
    Load all .txt files from *data_dir*, apply text preprocessing,
    and return a list of document dicts with keys:
        source_file, text, regulation_number, document_title, language
    """
    docs = []
    for path in sorted(Path(data_dir).glob("*.txt")):
        text = path.read_text(encoding="utf-8-sig")   # strips UTF-8 BOM
        text = text.replace("\u00AD", "")              # soft hyphen
        text = text.replace("\u00A0", " ")             # non-breaking space
        reg_num, title = parse_filename(path)
        docs.append({
            "source_file":       path.name,
            "text":              text,
            "regulation_number": reg_num,
            "document_title":    title,
            "language":          language,
        })
    return docs


def extract_section_id(text):
    """
    Parse the section heading from the start of a chunk text.
    Returns (section_id, section_title) or (None, None) if no heading found.
    """
    m = _HEADING_PARSE_RE.match(text.lstrip("\n"))
    if not m:
        return None, None
    return m.group(1), m.group(2).strip()


def split_with_overlap(text, max_chars, overlap):
    """
    Split *text* at paragraph boundaries (\n\n) so each piece is <= max_chars.
    The last *overlap* characters of each piece are prepended to the next one.
    """
    chunks = []
    start  = 0
    while start < len(text):
        end = start + max_chars
        if end >= len(text):
            chunks.append(text[start:])
            break
        # prefer paragraph break
        split_pos = text.rfind("\n\n", start, end)
        if split_pos <= start:
            split_pos = text.rfind("\n", start, end)
        if split_pos <= start:
            split_pos = end

        chunks.append(text[start:split_pos])
        # carry overlap, but always advance by at least 1 char
        new_start = max(split_pos - overlap, start + 1)
        start = new_start

    return [c for c in chunks if c.strip()]


def chunk_document(doc):
    """
    Split a document dict into a list of chunk text strings using a
    three-pass algorithm:
      Pass 1 — split on section headings
      Pass 2 — merge stubs < CHUNK_MIN_CHARS into the preceding section
      Pass 3 — re-split sections > CHUNK_MAX_CHARS at paragraph breaks
    """
    text = doc["text"]

    # Pass 1: split at section heading boundaries
    raw_parts = re.split(SECTION_HEADING_RE, text)

    # Pass 2: merge short stubs upward
    pending = ""
    merged  = []
    for part in raw_parts:
        combined = pending + part
        if len(combined.strip()) < CHUNK_MIN_CHARS:
            pending = combined
        else:
            merged.append(combined)
            pending = ""
    # flush remaining stub: attach to last chunk or keep as-is
    if pending.strip():
        if merged:
            merged[-1] += pending
        else:
            merged.append(pending)

    # Pass 3: re-split oversized chunks
    final = []
    for chunk in merged:
        if len(chunk) > CHUNK_MAX_CHARS:
            final.extend(split_with_overlap(chunk, CHUNK_MAX_CHARS, CHUNK_OVERLAP_CHARS))
        else:
            final.append(chunk)

    return final


def build_chunk_records(documents):
    """
    Convert a list of document dicts into a flat list of chunk dicts ready
    for ChromaDB.  Follower chunks (produced by split_with_overlap, no heading
    of their own) carry forward the last known section_id and section_title in
    metadata; the chunk text itself is NOT modified.

    Metadata fields per record:
        text, source_file, document_title, regulation_number,
        section_id, section_title, chunk_index, language
    """
    records = []
    for doc in documents:
        chunks = chunk_document(doc)
        last_section_id    = None
        last_section_title = None
        for i, chunk_text in enumerate(chunks):
            sid, stitle = extract_section_id(chunk_text)
            if sid is not None:
                last_section_id    = sid
                last_section_title = stitle
            records.append({
                "text":              chunk_text,
                "source_file":       doc["source_file"],
                "document_title":    doc["document_title"],
                "regulation_number": doc["regulation_number"],
                "section_id":        last_section_id,
                "section_title":     last_section_title,
                "chunk_index":       i,
                "language":          doc["language"],
            })
    return records


# ---------------------------------------------------------------------------
# Quick self-test (run as script: python chunking.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from pathlib import Path

    base = Path(__file__).parent / "data"
    all_docs = []
    for lang in ("de", "fr", "it"):
        lang_dir = base / lang
        if lang_dir.exists():
            docs = load_documents(lang_dir, lang)
            all_docs.extend(docs)
            print(f"  {lang}: {len(docs)} documents")

    records = build_chunk_records(all_docs)
    langs = {}
    for r in records:
        langs[r["language"]] = langs.get(r["language"], 0) + 1
    print(f"\nTotal chunks: {len(records)}")
    for lang, count in sorted(langs.items()):
        print(f"  {lang}: {count} chunks")
