import hashlib
import os
import re
import tempfile
from pathlib import Path

import faiss
import gdown
import numpy as np
import streamlit as st
from docx import Document
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# -----------------------------
# App settings
# -----------------------------

st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="📚",
    layout="wide",
)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 5

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "how", "i", "in", "is", "it", "of", "on", "or", "that", "the",
    "this", "to", "was", "what", "when", "where", "which", "who",
    "why", "with", "you", "your", "about", "can", "does", "do",
}


# -----------------------------
# Session state
# -----------------------------

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "index" not in st.session_state:
    st.session_state.index = None

if "document_hash" not in st.session_state:
    st.session_state.document_hash = None

if "document_info" not in st.session_state:
    st.session_state.document_info = []

if "source_files" not in st.session_state:
    st.session_state.source_files = []


# -----------------------------
# Embedding model
# -----------------------------

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


# -----------------------------
# Extraction functions
# -----------------------------

def extract_pdf(file_path, filename):
    """Extract one record per PDF page."""
    records = []
    reader = PdfReader(file_path)

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()

        if text:
            records.append({
                "filename": filename,
                "page": page_number,
                "text": text,
            })

    return records


def extract_docx(file_path, filename):
    """Extract text from a DOCX file."""
    document = Document(file_path)
    paragraphs = [p.text.strip() for p in document.paragraphs if p.text.strip()]
    text = "\n".join(paragraphs)

    if not text:
        return []

    return [{
        "filename": filename,
        "page": None,
        "text": text,
    }]


def extract_txt(file_path, filename):
    """Extract text from a TXT file."""
    text = Path(file_path).read_text(encoding="utf-8", errors="ignore").strip()

    if not text:
        return []

    return [{
        "filename": filename,
        "page": None,
        "text": text,
    }]


def extract_md(file_path, filename):
    """Extract text from a Markdown file."""
    text = Path(file_path).read_text(encoding="utf-8", errors="ignore").strip()

    if not text:
        return []

    return [{
        "filename": filename,
        "page": None,
        "text": text,
    }]


def extract_document(file_path, filename):
    """Choose the correct extraction function from the file extension."""
    extension = Path(filename).suffix.lower()

    if extension == ".pdf":
        return extract_pdf(file_path, filename)
    if extension == ".docx":
        return extract_docx(file_path, filename)
    if extension == ".txt":
        return extract_txt(file_path, filename)
    if extension == ".md":
        return extract_md(file_path, filename)

    return []


# -----------------------------
# Chunking
# -----------------------------

def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping character chunks."""
    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end].strip())

        if end >= len(text):
            break

        start = end - overlap

    return [chunk for chunk in chunks if chunk]


def create_chunks(records):
    """Create chunks while preserving filename and page metadata."""
    chunks = []

    for record in records:
        text_chunks = split_text(record["text"])

        for chunk in text_chunks:
            chunks.append({
                "filename": record["filename"],
                "page": record["page"],
                "text": chunk,
            })

    return chunks


# -----------------------------
# Embeddings + FAISS
# -----------------------------

def build_vector_index(chunks):
    """Create embeddings once and store them in a FAISS index."""
    if not chunks:
        return None

    model = load_embedding_model()
    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype("float32")

    # Normalize so inner product behaves like cosine similarity.
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index


# -----------------------------
# Keyword search
# -----------------------------

def important_words(question):
    words = re.findall(r"\b[a-zA-Z0-9_]+\b", question.lower())
    return [word for word in words if word not in STOP_WORDS and len(word) > 2]


def keyword_score(question, text):
    """Simple keyword overlap score between 0 and 1."""
    query_words = set(important_words(question))

    if not query_words:
        return 0.0

    text_words = set(re.findall(r"\b[a-zA-Z0-9_]+\b", text.lower()))
    matches = query_words.intersection(text_words)

    return len(matches) / len(query_words)


# -----------------------------
# Hybrid search
# -----------------------------

def hybrid_search(question, top_k=TOP_K):
    """Combine FAISS semantic search and keyword matching."""
    chunks = st.session_state.chunks
    index = st.session_state.index

    if not chunks or index is None:
        return []

    model = load_embedding_model()

    query_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype("float32")

    faiss.normalize_L2(query_embedding)

    # Search more candidates than the final result count so keyword
    # matching can influence the final ranking.
    candidate_k = min(max(top_k * 3, 10), len(chunks))
    semantic_scores, indices = index.search(query_embedding, candidate_k)

    candidates = []

    for semantic_score, chunk_index in zip(semantic_scores[0], indices[0]):
        if chunk_index < 0:
            continue

        chunk = chunks[int(chunk_index)].copy()

        # FAISS uses cosine similarity because vectors are normalized.
        semantic_score = float(semantic_score)

        # Convert roughly from [-1, 1] to [0, 1].
        semantic_normalized = max(0.0, min(1.0, (semantic_score + 1.0) / 2.0))

        keyword = keyword_score(question, chunk["text"])

        # Semantic search is weighted more heavily.
        hybrid = (0.70 * semantic_normalized) + (0.30 * keyword)

        chunk["semantic_score"] = semantic_normalized
        chunk["keyword_score"] = keyword
        chunk["hybrid_score"] = hybrid

        candidates.append(chunk)

    candidates.sort(key=lambda item: item["hybrid_score"], reverse=True)

    return candidates[:top_k]


# -----------------------------
# Google Drive
# -----------------------------

def load_drive_files(url):
    """
    Download a public Google Drive file or folder.

    The Drive item must be shared as "Anyone with the link".
    """
    temp_dir = Path(tempfile.mkdtemp(prefix="drive_docs_"))

    is_folder = "/folders/" in url

    if is_folder:
        gdown.download_folder(
            url=url,
            output=str(temp_dir),
            quiet=True,
            use_cookies=False,
        )
    else:
        output_file = temp_dir / "drive_download"
        downloaded = gdown.download(
            url=url,
            output=str(output_file),
            quiet=True,
            fuzzy=True,
        )

        if downloaded:
            downloaded_path = Path(downloaded)
            # gdown may determine the actual filename.
            if downloaded_path.exists():
                return [downloaded_path]

    files = []

    for path in temp_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path)

    return files


# -----------------------------
# Processing pipeline
# -----------------------------

def process_files(file_items):
    """
    Extract -> chunk -> embed -> FAISS.
    This is called only when the document set changes.
    """
    records = []
    document_info = []

    for item in file_items:
        path = item["path"]
        filename = item["filename"]

        try:
            extracted = extract_document(path, filename)
            records.extend(extracted)

            document_info.append({
                "filename": filename,
                "type": path.suffix.lower().replace(".", "").upper(),
                "pages": len(extracted) if path.suffix.lower() == ".pdf" else None,
                "characters": sum(len(record["text"]) for record in extracted),
            })
        except Exception as exc:
            st.warning(f"Could not read {filename}: {exc}")

    chunks = create_chunks(records)

    if not chunks:
        st.session_state.chunks = []
        st.session_state.index = None
        st.session_state.document_info = document_info
        return 0

    with st.spinner("Creating document embeddings..."):
        index = build_vector_index(chunks)

    st.session_state.chunks = chunks
    st.session_state.index = index
    st.session_state.document_info = document_info

    return len(chunks)


def make_document_hash(file_items):
    """Create a stable hash so unchanged documents are not embedded again."""
    hasher = hashlib.sha256()

    for item in sorted(file_items, key=lambda x: x["filename"]):
        hasher.update(item["filename"].encode("utf-8"))
        hasher.update(item["data"])

    return hasher.hexdigest()


# -----------------------------
# Groq
# -----------------------------

def get_groq_client():
    if "GROQ_API_KEY" not in st.secrets:
        return None

    api_key = st.secrets["GROQ_API_KEY"]

    if not api_key:
        return None

    return Groq(api_key=api_key)


def answer_question(question, retrieved_chunks):
    """Ask Groq to answer only from the retrieved context."""
    client = get_groq_client()

    if client is None:
        return "GROQ_API_KEY is not configured in Streamlit secrets."

    context_parts = []

    for number, chunk in enumerate(retrieved_chunks, start=1):
        page = f", page {chunk['page']}" if chunk["page"] else ""
        context_parts.append(
            f"[Source {number}: {chunk['filename']}{page}]\n"
            f"{chunk['text']}"
        )

    context = "\n\n".join(context_parts)

    system_prompt = """
You are an AI document assistant.

Answer the user's question ONLY using the provided document context.

Rules:
1. Do not use outside knowledge.
2. Do not invent facts.
3. If the answer is not available in the context, say:
   "I couldn't find that information in the provided documents."
4. Keep the answer clear and concise.
5. You may combine information from multiple provided chunks when needed.
"""

    user_prompt = f"""
DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )

    return response.choices[0].message.content


# -----------------------------
# UI helpers
# -----------------------------

def add_uploaded_files(uploaded_files):
    items = []

    for uploaded_file in uploaded_files:
        suffix = Path(uploaded_file.name).suffix.lower()

        if suffix not in SUPPORTED_EXTENSIONS:
            continue

        data = uploaded_file.getvalue()

        temp_dir = Path(tempfile.mkdtemp(prefix="local_docs_"))
        path = temp_dir / uploaded_file.name
        path.write_bytes(data)

        items.append({
            "filename": uploaded_file.name,
            "path": path,
            "data": data,
        })

    return items


def add_drive_files(paths):
    items = []

    for path in paths:
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        items.append({
            "filename": path.name,
            "path": path,
            "data": path.read_bytes(),
        })

    return items


# -----------------------------
# UI
# -----------------------------

st.title("📚 AI Document Assistant")
st.write(
    "Upload documents or load public Google Drive files, then ask questions "
    "using semantic + keyword search."
)

with st.sidebar:
    st.header("Document Sources")

    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, TXT or MD files",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
    )

    st.divider()

    drive_url = st.text_input(
        "Google Drive file/folder link",
        placeholder="https://drive.google.com/...",
    )

    load_drive = st.button("Load from Google Drive", use_container_width=True)

    if load_drive:
        if not drive_url.strip():
            st.warning("Paste a Google Drive file or folder link first.")
        else:
            with st.spinner("Downloading Google Drive files..."):
                try:
                    drive_paths = load_drive_files(drive_url.strip())

                    if drive_paths:
                        drive_items = add_drive_files(drive_paths)
                        st.session_state.source_files.extend(drive_items)
                        st.success(f"Loaded {len(drive_items)} supported file(s).")
                    else:
                        st.warning(
                            "No supported PDF, DOCX, TXT or MD files were found."
                        )
                except Exception as exc:
                    st.error(f"Google Drive download failed: {exc}")

# Add local uploads to the current source list.
if uploaded_files:
    new_uploads = add_uploaded_files(uploaded_files)

    existing_names = {
        item["filename"] for item in st.session_state.source_files
    }

    for item in new_uploads:
        if item["filename"] not in existing_names:
            st.session_state.source_files.append(item)

# Process only when source documents actually change.
if st.session_state.source_files:
    current_hash = make_document_hash(st.session_state.source_files)

    if current_hash != st.session_state.document_hash:
        with st.spinner("Extracting and processing documents..."):
            chunk_count = process_files(st.session_state.source_files)

        st.session_state.document_hash = current_hash
        st.success(f"Documents processed. Created {chunk_count} chunks.")

if st.session_state.document_info:
    st.subheader("Document Information")

    for info in st.session_state.document_info:
        page_text = (
            f"{info['pages']} PDF pages"
            if info["pages"] is not None
            else "Page number not available"
        )

        st.write(
            f"**{info['filename']}** — {info['type']} — "
            f"{page_text} — {info['characters']:,} extracted characters"
        )

    st.info(
        f"Total chunks: {len(st.session_state.chunks)} | "
        f"Embeddings are reused for questions during this session."
    )

st.divider()

st.subheader("Ask a Question")

question = st.text_input(
    "Your question",
    placeholder="What does the document say about...?",
)

if question:
    if not st.session_state.chunks:
        st.warning("Add at least one supported document first.")
    else:
        with st.spinner("Searching documents..."):
            results = hybrid_search(question)

        if not results:
            st.warning("No relevant document chunks were found.")
        else:
            with st.spinner("Generating answer..."):
                answer = answer_question(question, results)

            st.markdown("### Answer")
            st.write(answer)

            st.markdown("### Retrieved Sources")

            for number, result in enumerate(results, start=1):
                page = (
                    f"Page {result['page']}"
                    if result["page"]
                    else "Page not available"
                )

                with st.expander(
                    f"{number}. {result['filename']} — {page} "
                    f"(score: {result['hybrid_score']:.3f})"
                ):
                    st.caption(
                        f"Semantic: {result['semantic_score']:.3f} | "
                        f"Keyword: {result['keyword_score']:.3f}"
                    )
                    st.write(result["text"])

st.caption(
    "Pipeline: Extract → Chunk → Embed once → FAISS + Keyword Hybrid Search → Groq"
)
