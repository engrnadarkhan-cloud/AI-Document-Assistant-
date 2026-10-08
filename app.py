import streamlit as st
import os, re, uuid, hashlib, shutil, time
import numpy as np, pandas as pd
import fitz, docx, faiss
from PIL import Image
import io, pytesseract
from rank_bm25 import BM25Okapi
import gdown
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import google.generativeai as genai

# App Page Setup
st.set_page_config(page_title="PSX Investor Intelligence", layout="wide", page_icon="📈")

# ==========================================
# 1. DATA STRUCTURE
# ==========================================
@dataclass
class DocumentChunk:
    chunk_id: str
    text: str
    metadata: Dict[str, Any]

@dataclass
class RetrievalResult:
    chunk: DocumentChunk
    semantic_score: float = 0.0
    semantic_rank: int = 0
    keyword_score: float = 0.0
    keyword_rank: int = 0
    rrf_score: float = 0.0
    quality_flag: str = "GOOD"

# ==========================================
# 2. FILE DOWNLOAD AUR EXTRACTION
# ==========================================
def ingest_chain(drive_url: str, output_dir: str = "data") -> str:
    match = re.search(r"folders/([a-zA-Z0-9_-]+)", drive_url)
    if not match: match = re.search(r"id=([a-zA-Z0-9_-]+)", drive_url)
    if not match: raise ValueError("Invalid Google Drive Folder URL.")
    
    if os.path.exists(output_dir): shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    gdown.download_folder(id=match.group(1), output=output_dir, quiet=True, use_cookies=False)
    return output_dir

def extract_chain(directory: str) -> List[Dict[str, Any]]:
    raw_docs = []
    for root, _, files in os.walk(directory):
        for file in files:
            filepath = os.path.join(root, file)
            ext = file.lower().split('.')[-1]
            doc_hash = hashlib.sha256(open(filepath, 'rb').read()).hexdigest()
            
            if ext == "pdf":
                doc = fitz.open(filepath)
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    text = page.get_text().strip()
                    method = "native_text"
                    
                    if len(text) < 50:
                        pix = page.get_pixmap()
                        img = Image.open(io.BytesIO(pix.tobytes("png")))
                        text = pytesseract.image_to_string(img).strip()
                        method = "ocr"
                        
                    if len(text) > 20:
                        raw_docs.append({"text": text, "metadata": {"document_name": file, "file_type": "pdf", "page_number": page_num + 1, "extraction_method": method, "document_hash": doc_hash}})
            elif ext in ["xlsx", "xls"]:
                xl = pd.ExcelFile(filepath)
                for sheet in xl.sheet_names:
                    df = xl.parse(sheet).dropna(how="all")
                    if not df.empty:
                        text = f"Sheet Name: {sheet}\n" + df.to_csv(index=False)
                        raw_docs.append({"text": text, "metadata": {"document_name": file, "file_type": "xlsx", "sheet_name": sheet, "extraction_method": "native_text", "document_hash": doc_hash}})
    return raw_docs

def metadata_chain(normalized_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for doc in normalized_docs:
        filename = doc["metadata"]["document_name"].upper()
        year_match = re.search(r'(20\d{2})', filename)
        doc["metadata"]["fiscal_year"] = int(year_match.group(1)) if year_match else None
        
        if "ANNUAL" in filename or "AR" in filename: doc["metadata"]["document_type"] = "Annual Report"
        elif "Q1" in filename or "Q2" in filename or "Q3" in filename or "QUARTER" in filename: doc["metadata"]["document_type"] = "Quarterly Report"
        else: doc["metadata"]["document_type"] = "Other"
    return normalized_docs

def chunk_chain(docs: List[Dict[str, Any]], chunk_size: int = 600, overlap: int = 100) -> List[DocumentChunk]:
    chunks = []
    for doc in docs:
        words = doc["text"].split()
        for i in range(0, max(1, len(words)), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            if len(chunk_words) > 10:
                chunks.append(DocumentChunk(chunk_id=str(uuid.uuid4()), text=" ".join(chunk_words), metadata=doc["metadata"].copy()))
    return chunks

# ==========================================
# 3. KNOWLEDGE BASE (Google Generative AI SDK)
# ==========================================
def embedding_chain(chunks: List[DocumentChunk], api_key: str) -> Tuple[List[DocumentChunk], np.ndarray]:
    genai.configure(api_key=api_key)
    valid_chunks = []
    all_vectors = []
    
    # 👈 Yahan new model update ho gaya hai!
    embedding_model = "models/gemini-embedding-001"
    batch_size = 20
    
    texts = [c.text for c in chunks if c.text and len(c.text.strip()) >= 10]
    chunk_map = [c for c in chunks if c.text and len(c.text.strip()) >= 10]
    
    if not texts:
        raise ValueError("Documents se koi valid text nahi nikla.")
        
    total_batches = (len(texts) + batch_size - 1) // batch_size
    progress_bar = st.progress(0, text=f"AI Embeddings ban rahi hain (0/{total_batches} batches)...")
    
    last_error = ""
    error_count = 0
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_chunks = chunk_map[i:i+batch_size]
        
        try:
            res = genai.embed_content(
                model=embedding_model,
                content=batch_texts,
                task_type="retrieval_document"
            )
            vecs = res['embedding']
            for chunk, vec in zip(batch_chunks, vecs):
                valid_chunks.append(chunk)
                all_vectors.append(np.array(vec, dtype=np.float32))
            error_count = 0
            time.sleep(0.5)
        except Exception as e:
            last_error = str(e)
            error_count += 1
            if error_count >= 3:
                break
            time.sleep(2)
            continue
            
        progress_bar.progress(min((i + batch_size) / len(texts), 1.0))
        
    progress_bar.empty()
    
    if not all_vectors:
        st.error(f"🚨 API Error:\n`{last_error}`\n(Please check Gemini API limits)")
        st.stop()
        
    final_vectors = np.vstack(all_vectors)
    faiss.normalize_L2(final_vectors)
    return valid_chunks, final_vectors

def indexing_chain(chunks: List[DocumentChunk], vectors: np.ndarray) -> Tuple[faiss.IndexFlatIP, BM25Okapi]:
    dimension = vectors.shape[1]
    faiss_index = faiss.IndexFlatIP(dimension)
    faiss_index.add(vectors)
    tokenized_corpus = [c.text.lower().split() for c in chunks]
    bm25_index = BM25Okapi(tokenized_corpus)
    return faiss_index, bm25_index

# ==========================================
# 4. SEARCH AUR REASONING
# ==========================================
def hybrid_retrieval_chain(query: str, chunks: List[DocumentChunk], faiss_idx: faiss.IndexFlatIP, bm25_idx: BM25Okapi, api_key: str) -> List[RetrievalResult]:
    genai.configure(api_key=api_key)
    
    # 👈 Yahan bhi new model update ho gaya hai!
    q_res = genai.embed_content(
        model="models/gemini-embedding-001",
        content=query,
        task_type="retrieval_query"
    )
        
    q_vec = np.array([q_res['embedding']], dtype=np.float32)
    faiss.normalize_L2(q_vec)
    sem_scores, sem_indices = faiss_idx.search(q_vec, len(chunks))
    
    tokenized_query = query.lower().split()
    bm25_scores = bm25_idx.get_scores(tokenized_query)
    bm25_indices = np.argsort(bm25_scores)[::-1]

    results_map = {idx: RetrievalResult(chunk=chunks[idx]) for idx in range(len(chunks))}
    for rank, idx in enumerate(sem_indices[0]):
        results_map[idx].semantic_score = float(sem_scores[0][rank])
        results_map[idx].semantic_rank = rank + 1
    for rank, idx in enumerate(bm25_indices):
        results_map[idx].keyword_score = float(bm25_scores[idx])
        results_map[idx].keyword_rank = rank + 1
        
    for res in results_map.values():
        sem_rrf = 1.0 / (60 + res.semantic_rank) if res.semantic_score > 0 else 0
        kw_rrf = 1.0 / (60 + res.keyword_rank) if res.keyword_score > 0 else 0
        res.rrf_score = sem_rrf + kw_rrf

    sorted_results = sorted(list(results_map.values()), key=lambda x: x.rrf_score, reverse=True)
    return sorted_results[:7]

def final_answer_chain(query: str, evidence: List[RetrievalResult], api_key: str) -> str:
    if not evidence: return "Barae meharbani is sawal ke liye mazeed wazeh data Drive mein dalein."
    
    genai.configure(api_key=api_key)
    
    context = ""
    for i, res in enumerate(evidence):
        context += f"--- [S{i+1}] {res.chunk.metadata.get('document_name')} (Page {res.chunk.metadata.get('page_number', 'N/A')}) ---\n{res.chunk.text}\n\n"
        
    prompt = f"""You are a PSX Investment Analyst. Use ONLY the evidence provided to answer.
1. Never fabricate financial figures. Preserve exact units.
2. Cite all claims inline using [S1], [S2].
USER QUERY: {query}
EVIDENCE CORPUS:
{context}"""
    
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    reasoning = response.text
    
    citations = "\n\n---\n### 📚 Traceable Sources\n"
    for i, res in enumerate(evidence):
        citations += f"* **[S{i+1}]** {res.chunk.metadata.get('document_name')} (Page {res.chunk.metadata.get('page_number', 'N/A')})\n"
    
    return reasoning + citations

# ==========================================
# 5. UI (User Interface)
# ==========================================
def main():
    st.title("📊 PSX Intelligence Engine")
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        drive_url = st.text_input("Drive Folder URL")
        chunk_size = st.number_input("Chunk Size", value=600)
        chunk_overlap = st.number_input("Overlap", value=100)
        
        if st.button("🚀 Knowledge Base Banayein", type="primary"):
            api_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
            if not api_key: 
                st.error("Missing GEMINI_API_KEY in Secrets.")
            elif not drive_url: 
                st.error("Please enter Drive URL.")
            else:
                with st.status("Data parha ja raha hai...", expanded=True) as status:
                    st.write("1. Drive se download ho raha hai...")
                    data_dir = ingest_chain(drive_url)
                    
                    downloaded_files = []
                    for root, _, files in os.walk(data_dir):
                        downloaded_files.extend(files)
                    
                    if not downloaded_files:
                        st.error("🚨 Google Drive se 0 files download hui hain!")
                        st.stop()
                        
                    st.write(f"📁 Drive se {len(downloaded_files)} files mil gayin!")
                    st.write("2. Text aur Tables nikal rahe hain...")
                    raw = extract_chain(data_dir)
                    
                    if not raw:
                        st.error("🚨 Files mili hain, lekin unme parhne layeq text nahi hai.")
                        st.stop()
                        
                    st.write("3. Saal aur Metadata set ho raha hai...")
                    meta = metadata_chain(raw)
                    
                    st.write(f"4. Chunks ban rahe hain (Size: {chunk_size})...")
                    raw_chunks = chunk_chain(meta, int(chunk_size), int(chunk_overlap))
                    st.write(f"👉 Total **{len(raw_chunks)} chunks** ban gaye hain.")
                    
                    st.write("5. AI dimaagh ban raha hai (Indexing)...")
                    valid_chunks, vectors = embedding_chain(raw_chunks, api_key)
                    faiss_idx, bm25_idx = indexing_chain(valid_chunks, vectors)
                    
                    st.session_state["db"] = {"chunks": valid_chunks, "faiss": faiss_idx, "bm25": bm25_idx}
                    status.update(label="Knowledge Base Taiyar Hai!", state="complete")

    if "db" in st.session_state:
        db = st.session_state["db"]
        st.success(f"Total {len(db['chunks'])} hisson (chunks) ka data successfully parh liya gaya hai.")
        
        query = st.text_input("Apna Sawal likhein (Misaal: '2024 aur 2025 ke EPS ko compare karein'):")
        
        if st.button("Jawab Dhoondein") and query:
            api_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
            with st.spinner("AI saboot dhoond raha hai..."):
                results = hybrid_retrieval_chain(query, db["chunks"], db["faiss"], db["bm25"], api_key)
                answer = final_answer_chain(query, results, api_key)
                st.markdown(answer)

if __name__ == "__main__":
    main()
