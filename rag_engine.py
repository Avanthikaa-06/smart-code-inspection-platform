import os
import re
import pickle
import numpy as np

def _load_html(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        html = f.read()
    html = re.sub(r"<script.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<style.*?</style>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).strip()

def _load_pdf(path: str) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        raise RuntimeError("pypdf is not installed. Run: pip install pypdf")
    
    reader, text_parts = PdfReader(path), []
    for page in reader.pages:
        try: text_parts.append(page.extract_text() or "")
        except Exception: text_parts.append("")
    return "\n".join(text_parts)

def _load_docx(path: str) -> str:
    try:
        import docx
    except ImportError:
        raise RuntimeError("python-docx is not installed. Run: pip install python-docx")
    return "\n".join(p.text for p in docx.Document(path).paragraphs)

def _load_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

SUPPORTED_LOADERS = {".pdf": _load_pdf, ".txt": _load_txt, ".docx": _load_docx, ".html": _load_html, ".htm": _load_html}

def load_documents(knowledge_base_dir: str = "knowledge_base") -> list:
    documents = []
    if not os.path.isdir(knowledge_base_dir): return documents
    
    for fname in sorted(os.listdir(knowledge_base_dir)):
        loader = SUPPORTED_LOADERS.get(os.path.splitext(fname)[1].lower())
        if loader:
            try:
                text = loader(os.path.join(knowledge_base_dir, fname))
                if text and text.strip(): documents.append({"source": fname, "text": text})
            except Exception as e:
                print(f"[rag_engine] Skipped '{fname}': {e}")
    return documents

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    if not text or not text.strip(): return []
    text = text.strip()
    if chunk_size <= 0: return [text]
    
    step, chunks, start, text_len = max(chunk_size - overlap, 1), [], 0, len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()
        if chunk: chunks.append(chunk)
        if end == text_len: break
        start += step
    return chunks

def chunk_documents(documents: list) -> list:
    return [{"source": doc.get("source", "unknown"), "chunk_id": idx, "text": piece}
            for doc in documents for idx, piece in enumerate(chunk_text(doc.get("text", "")))]

MODEL_NAME = "all-MiniLM-L6-v2"
_model = None

def get_model():
    global _model
    if _model is not None: return _model
    try:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
        return _model
    except ImportError:
        raise RuntimeError("sentence-transformers is not installed. Run: pip install sentence-transformers")
    except Exception as e:
        raise RuntimeError(f"Could not load embedding model '{MODEL_NAME}': {e}")

def embed_texts(texts: list):
    return get_model().encode(texts, show_progress_bar=False, convert_to_numpy=True) if texts else []

def embed_query(query: str):
    return get_model().encode([query], show_progress_bar=False, convert_to_numpy=True)[0] if query and query.strip() else None

class VectorStore:
    def __init__(self, dimension: int = None):
        self.dimension, self.index, self.metadata = dimension, None, []

    def build(self, embeddings, metadata: list):
        try:
            import faiss
        except ImportError:
            raise RuntimeError("faiss-cpu is not installed. Run: pip install faiss-cpu")
            
        if embeddings is None or len(embeddings) == 0:
            self.index, self.metadata = None, []
            return
            
        embeddings = np.asarray(embeddings, dtype="float32")
        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings)
        self.metadata = metadata

    def search(self, query_embedding, top_k: int = 3) -> list:
        if self.index is None or query_embedding is None: return []
        distances, indices = self.index.search(np.asarray([query_embedding], dtype="float32"), top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if 0 <= idx < len(self.metadata):
                record = dict(self.metadata[idx])
                record["score"] = float(dist)
                results.append(record)
        return results

    def is_ready(self) -> bool: return self.index is not None and len(self.metadata) > 0

    def save(self, directory: str):
        try: import faiss
        except ImportError: raise RuntimeError("faiss-cpu is not installed. Run: pip install faiss-cpu")
        os.makedirs(directory, exist_ok=True)
        if self.index is not None: faiss.write_index(self.index, os.path.join(directory, "index.faiss"))
        with open(os.path.join(directory, "metadata.pkl"), "wb") as f: pickle.dump(self.metadata, f)

    def load(self, directory: str) -> bool:
        idx_p, meta_p = os.path.join(directory, "index.faiss"), os.path.join(directory, "metadata.pkl")
        if not os.path.exists(idx_p) or not os.path.exists(meta_p): return False
        try:
            import faiss
            self.index = faiss.read_index(idx_p)
            with open(meta_p, "rb") as f: self.metadata = pickle.load(f)
            return True
        except Exception: return False

CACHE_DIR = os.path.join(os.path.dirname(__file__), ".rag_cache")

class RAGPipeline:
    def __init__(self, knowledge_base_dir: str = "knowledge_base"):
        self.knowledge_base_dir, self.store, self.num_documents, self.num_chunks, self.error = knowledge_base_dir, VectorStore(), 0, 0, None

    def build_index(self, use_cache: bool = True, api_key: str = None) -> dict:
        self.error = None
        if use_cache and self.store.load(CACHE_DIR) and self.store.is_ready():
            self.num_chunks = len(self.store.metadata)
            self.num_documents = len({m["source"] for m in self.store.metadata})
            return self._status(cached=True)

        try:
            documents = load_documents(self.knowledge_base_dir)
            self.num_documents = len(documents)
            if not documents:
                self.error = f"No valid documents found in '{self.knowledge_base_dir}/'."
                return self._status(cached=False)

            chunks = chunk_documents(documents)
            self.num_chunks = len(chunks)
            if not chunks:
                self.error = "Documents were found but no text could be extracted."
                return self._status(cached=False)

            self.store.build(embed_texts([c["text"] for c in chunks]), chunks)
            try: self.store.save(CACHE_DIR)
            except Exception: pass
            return self._status(cached=False)
        except Exception as e:
            self.error = str(e)
            return self._status(cached=False)

    def _status(self, cached: bool) -> dict:
        return {"ready": self.store.is_ready(), "num_documents": self.num_documents, "num_chunks": self.num_chunks, "cached": cached, "error": self.error}

    def query(self, question: str, top_k: int = 3) -> dict:
        if not question or not question.strip(): return {"success": False, "error": "Please enter a question.", "results": []}
        if not self.store.is_ready(): return {"success": False, "error": "The knowledge base index has not been built yet.", "results": []}
        try: return {"success": True, "error": None, "results": self.store.search(embed_query(question), top_k=top_k)}
        except Exception as e: return {"success": False, "error": str(e), "results": []}