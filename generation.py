"""
Answer generation and language handling for the FDV RAG system.
Imported by all RAG notebooks.
"""

import ollama
from langdetect import detect, LangDetectException
from langdetect import DetectorFactory

from retrieval import retrieve

# Seed langdetect for reproducibility (Italian short queries can be mis-detected)
DetectorFactory.seed = 42

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert on Swiss Railway operating regulations (FDV - Fahrdienstvorschriften).\n"
    "Answer ONLY from the provided excerpts.\n"
    "Cite section numbers and document names when possible.\n"
    "If the provided context is insufficient to answer the question, say so clearly."
)

LANG_NAMES = {
    "it": "Italian",
    "fr": "French",
    "de": "German",
    "en": "English",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_language(query: str) -> str:
    """
    Detect the language of *query* and return the language name string
    (e.g. "German").  Falls back to "the same language as the question"
    when detection fails or returns an unsupported code.
    """
    try:
        code = detect(query)
        return LANG_NAMES.get(code, "the same language as the question")
    except LangDetectException:
        return "the same language as the question"


def build_user_message(query: str, chunks: list, lang_name: str) -> str:
    """
    Format retrieved chunks + query into the user turn sent to the LLM.
    """
    parts = []
    for i, chunk in enumerate(chunks, 1):
        title  = chunk.get("document_title", "")
        sec_id = chunk.get("section_id") or ""
        sec_tt = chunk.get("section_title") or ""
        header = f"[Excerpt {i} \u2014 {title}, Section {sec_id}: {sec_tt}]"
        parts.append(f"{header}\n{chunk['text'].strip()}")

    excerpts_block = "\n---\n".join(parts)
    return (
        f"{excerpts_block}\n"
        f"---\n"
        f"QUESTION: {query}\n"
        f"IMPORTANT: You must write your entire answer in {lang_name} only.\n"
        f"Do not use any other language."
    )


def ask(query, embedder, cross_encoder, collection, bm25_index, chunk_records, cfg):
    """
    Full RAG pipeline: retrieve relevant chunks, build prompt, call LLM.

    Returns
    -------
    (answer_str, retrieved_chunks)
    """
    chunks    = retrieve(query, embedder, cross_encoder,
                         collection, bm25_index, chunk_records, cfg)
    lang_name = detect_language(query)
    user_msg  = build_user_message(query, chunks, lang_name)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_msg},
    ]

    try:
        response = ollama.chat(
            model=cfg["MODEL_NAME"],
            messages=messages,
            options={"temperature": 0.1},
        )
        answer = response.message.content
    except Exception as exc:
        answer = (
            f"[Ollama error: {exc}]\n"
            "Make sure Ollama is running (`ollama serve`) and the model is pulled "
            f"(`ollama pull {cfg.get('MODEL_NAME', 'phi4-mini')}`)."
        )

    return answer, chunks
