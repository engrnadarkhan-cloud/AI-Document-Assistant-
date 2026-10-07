# 📚 AI Document Assistant

A simple Streamlit-based AI Document Assistant that lets you upload documents, search them semantically and by keywords, and ask questions using Groq.

## Features

- Upload **PDF, DOCX, TXT and MD** files.
- Extract document text with separate extraction functions.
- Preserve:
  - filename
  - PDF page number when available
- Split documents into overlapping text chunks.
- Show the number of created chunks.
- Create **Sentence Transformers** embeddings for all chunks.
- Store the FAISS index and chunk metadata in Streamlit session state.
- Reuse embeddings for multiple questions instead of creating them again.
- Search with:
  - FAISS semantic similarity
  - simple keyword matching
- Combine both methods with a hybrid score.
- Send only retrieved document chunks to Groq.
- Instruct the model to answer only from the provided context.
- Show retrieved sources after every answer.
- Load public Google Drive files or folders.
- Google Drive files use the same extraction → chunking → embedding → search pipeline.
- Keep the Groq API key in Streamlit secrets.

## Project structure

```text
ai_document_assistant/
├── app.py
├── requirements.txt
└── readme.md
```

## 1. Create the project

```bash
mkdir ai_document_assistant
cd ai_document_assistant
```

Put the three files in this folder:

```text
app.py
requirements.txt
readme.md
```

## 2. Create a virtual environment

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

The first run may take some time because Sentence Transformers downloads the `all-MiniLM-L6-v2` embedding model.

## 4. Configure the Groq API key

Create:

```text
.streamlit/secrets.toml
```

Add:

```toml
GROQ_API_KEY = "your-groq-api-key"
```

Do **not** commit this file to GitHub.

Add this to `.gitignore`:

```text
.streamlit/secrets.toml
.venv/
__pycache__/
```

The application reads the key using:

```python
st.secrets["GROQ_API_KEY"]
```

The API key is never hardcoded in `app.py`.

## 5. Run the app

```bash
streamlit run app.py
```

Then open the Streamlit URL shown in the terminal.

## 6. Using local documents

Upload one or more:

- `.pdf`
- `.docx`
- `.txt`
- `.md`

The application will:

```text
Document
   ↓
Text extraction
   ↓
Overlapping chunks
   ↓
Sentence Transformer embeddings
   ↓
FAISS index
```

The embeddings are created when the document set changes.

When you ask another question, the application does **not** recreate all document embeddings.

Instead:

```text
Question
   ↓
Question embedding
   ↓
FAISS semantic search
   +
Keyword search
   ↓
Hybrid ranking
   ↓
Top document chunks
   ↓
Groq
   ↓
Answer + sources
```

## 7. Google Drive

Paste a Google Drive file or folder link into the sidebar.

The Drive item must be publicly accessible using:

```text
Anyone with the link
```

Supported files:

- PDF
- DOCX
- TXT
- MD

For folders, the app downloads supported files and sends them through the same document pipeline.

Example:

```text
Google Drive folder
       ↓
Download supported files
       ↓
Extract
       ↓
Chunk
       ↓
Embed
       ↓
FAISS + keyword search
       ↓
Groq
```

## Important Google Drive limitation

This simple version uses `gdown`, so Drive files/folders need to be accessible through a public share link.

It is not a full Google OAuth/Google Drive API integration.

For a private company Drive later, replace this part with Google Drive API + OAuth.

## 8. How hybrid search works

The application uses two simple signals.

### Semantic score

Sentence Transformers converts the question and document chunks into vectors.

FAISS finds chunks that are semantically similar to the question.

### Keyword score

The app extracts important words from the question and checks how many occur in each chunk.

### Combined score

The current simple weighting is:

```text
Hybrid score =
    70% semantic similarity
    +
    30% keyword matching
```

The highest-scoring chunks are sent to Groq.

## 9. Source information

After every answer, the app displays the retrieved chunks.

For PDF:

```text
filename.pdf — Page 4
```

For TXT/DOCX/MD:

```text
filename.docx — Page not available
```

The actual retrieved text is shown inside an expandable source section.

## 10. Security

Never put your API key directly inside:

```python
app.py
```

Bad:

```python
GROQ_API_KEY = "gsk_..."
```

Correct:

```toml
# .streamlit/secrets.toml
GROQ_API_KEY = "gsk_..."
```

And in Python:

```python
st.secrets["GROQ_API_KEY"]
```

For Streamlit Community Cloud, add the same secret in the application's **Secrets** settings instead of committing `secrets.toml`.

## 11. Simple explanation of the project

This application is a small RAG system.

RAG means:

**Retrieval-Augmented Generation**

It does not send the entire document to the AI model.

Instead, it:

1. Reads the documents.
2. Splits them into small chunks.
3. Converts chunks into embeddings.
4. Stores the embeddings in FAISS.
5. Searches for chunks related to the user's question.
6. Adds keyword matching to improve retrieval.
7. Sends only the best chunks to Groq.
8. Generates an answer from those chunks.
9. Shows the sources used for the answer.

This keeps the application relatively simple while demonstrating the main building blocks of a document-based AI assistant.
