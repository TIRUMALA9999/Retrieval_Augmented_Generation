# Retrieval-Augmented Generation (RAG) — End-to-End Portfolio (Qdrant • LlamaIndex • Ollama • Graph RAG • Multimodal)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![RAG](https://img.shields.io/badge/RAG-Retrieval%20Augmented%20Generation-purple)
![Qdrant](https://img.shields.io/badge/VectorDB-Qdrant-green)
![LlamaIndex](https://img.shields.io/badge/Framework-LlamaIndex-black)
![Ollama](https://img.shields.io/badge/LLM-Ollama-orange)

**Target audience:** Recruiters / Hiring Managers  
**Style:** Resume-focused • Interview-explainable • ATS-friendly

This repository is a collection of hands-on **RAG system implementations**—from classic document Q&A to **reranking**, **multimodal RAG (text+image)**, and **Graph RAG**—built primarily with **LlamaIndex**, **Qdrant**, and **local LLM inference via Ollama**.

---

## Key highlights (30-second recruiter summary)

- Implemented multiple **RAG architectures**: classic vector RAG, **reranked RAG**, **multimodal RAG**, and **Graph RAG**.
- Used **Qdrant** as the vector database for fast semantic retrieval and persisted local collections (`chat_with_docs`, `document_chat`).
- Built document ingestion pipelines with **chunking + embeddings** using **HuggingFace (BGE)** and **fastembed (CLIP)**.
- Added **reranking** (`cross-encoder/ms-marco-MiniLM-L-2-v2`) to improve retrieval precision before generation.
- Deployed a **Streamlit document-chat UI** (LLM + retrieval + PDF preview) using **LlamaIndex + Ollama**.
- Implemented **Graph RAG** using **Neo4jPropertyGraphStore** + LlamaIndex **PropertyGraphIndex** for structured retrieval.

**ATS keywords:** Retrieval Augmented Generation, RAG, Vector Database, Qdrant, Embeddings, Semantic Search, Reranking, LlamaIndex, Ollama, Graph RAG, Neo4j, Multimodal Retrieval, CLIP, Python, Streamlit.

---

## What’s inside (repo map)

> This repo contains notebooks + apps for multiple RAG variants. Large generated artifacts like `qdrant_storage/` and `hf_cache/` are included in this snapshot (good for demos, but should be `.gitignore`d for production).

```
Retrieval_Augmented_Generation-main/
├─ RAG.ipynb                          # Classic RAG with Qdrant + BGE embeddings + reranker + Ollama
├─ chat_with_docs.py                  # Streamlit doc-chat app (PDF viewer + RAG + Ollama)
│
├─ rag-project-2/rag-project/
│  ├─ notebook.ipynb                  # RAG pipeline (docs → Qdrant → query engine)
│  └─ app.py                          # Streamlit UI version of doc-chat
│
├─ Faster_RAG_System/notebook.ipynb    # Batched embedding + speed-oriented RAG experiments (SQuAD)
│  └─ hf_cache/                        # HF embedding cache (generated)
│
├─ rag-evaluation/Untitled.ipynb       # RAG evaluation experiments (retrieval + generation)
│
├─ multimodal_rag_systems/project.ipynb# Multimodal RAG (fastembed CLIP text+image embeddings + Qdrant)
│
├─ graph-rag/graph_rag.ipynb           # Graph RAG with Neo4jPropertyGraphStore + PropertyGraphIndex
│  └─ data/paul_graham_essay.txt       # Example corpus
│
├─ qdrant_storage/                     # Local Qdrant persisted collection (generated)
└─ paul_graham/                        # Example text corpus
```

---

## How to explain this project in an interview (simple architecture)

### 1) Classic Vector RAG (Qdrant + Embeddings + Top‑K Retrieval)
1. **Ingest documents** (e.g., PDFs/text)
2. **Chunk** content into passages
3. **Embed** each chunk (BGE / fastembed)
4. **Store** embeddings in Qdrant
5. **Retrieve** top‑k chunks for a query
6. **Generate** answer with LLM (Ollama), grounded in retrieved context

### 2) Reranked RAG (better precision)
After retrieving top‑k, apply a **cross‑encoder reranker** to reorder candidates and keep the best `top_n`.  
This improves answer relevance by sending higher-quality context to the LLM.

### 3) Multimodal RAG (Text + Image)
- Build separate **text and image embeddings** (CLIP text/vision via fastembed)
- Store vectors in Qdrant and retrieve the most relevant text/images for a query.

### 4) Graph RAG (Neo4j + PropertyGraphIndex)
- Extract entities/relations from documents into a **property graph**
- Retrieve context using graph traversals (paths, synonyms) and/or vector context retriever
- Generate a grounded answer using both **graph structure + text context**.

---

## Quickstart (run locally)

### Prerequisites
- **Python 3.10+**
- **Ollama** installed and running
- **Qdrant** running locally (Docker recommended)
- **Neo4j** (only for Graph RAG notebook)

### 1) Start Qdrant
```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

### 2) Start Ollama + pull a model
```bash
ollama serve
ollama pull gemma3:1b
# or: ollama pull llama3.2:1b
```

> Some notebooks/apps reference these small local models for fast iteration.

### 3) Create environment + install deps
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -U pip
pip install qdrant-client llama-index llama-index-vector-stores-qdrant llama-index-llms-ollama llama-index-embeddings-huggingface             sentence-transformers streamlit fastembed pillow nest-asyncio transformers torch datasets tqdm
```

### 4) Run the core RAG notebook
```bash
jupyter lab
# Open: RAG.ipynb
```

### 5) Run the Streamlit document-chat app
Option A:
```bash
streamlit run chat_with_docs.py
```

Option B:
```bash
streamlit run rag-project-2/rag-project/app.py
```

---

## Where recruiters should look first

1. **RAG.ipynb**  
   Shows: Qdrant ingestion → BGE embeddings → reranker → Ollama generation.

2. **chat_with_docs.py** (or `rag-project-2/rag-project/app.py`)  
   Shows: an end-user UI for document chat + PDF preview + caching.

3. **multimodal_rag_systems/project.ipynb**  
   Shows: CLIP embeddings (text+vision) + multimodal retrieval in Qdrant.

4. **graph-rag/graph_rag.ipynb**  
   Shows: Graph RAG using Neo4j property graph + LlamaIndex property graph index.

---

## Implementation notes (engineering maturity)

- **Persisted vector DB artifacts included:** `qdrant_storage/`  
  Helpful for demos, but in production you’d typically:
  - run Qdrant as a service
  - avoid committing database files
  - store embeddings in managed storage

- **HF cache included:** `hf_cache/`  
  Good for speeding up local runs; normally this should be excluded from Git.

Suggested `.gitignore` entries:
```gitignore
qdrant_storage/
**/qdrant_storage/
hf_cache/
**/hf_cache/
*.mmap
*.dat
*.wal
```

---

## Resume bullet points (copy/paste)

- Built end-to-end **Retrieval-Augmented Generation (RAG)** pipelines using **Qdrant** and **LlamaIndex**, enabling document-grounded Q&A and reducing hallucinations through retrieval-based grounding.  
- Implemented **HuggingFace BGE embeddings** and optimized retrieval quality using a **cross-encoder reranker** (`ms-marco-MiniLM`) to improve top‑k relevance before generation.  
- Developed a **Streamlit document-chat application** that supports PDF ingestion, semantic retrieval, and **local LLM inference via Ollama** for interactive knowledge assistants.  
- Prototyped **Multimodal RAG** using **fastembed CLIP** text/vision embeddings and Qdrant indexing to retrieve relevant text/images for queries.  
- Implemented **Graph RAG** using **Neo4j property graph store** and LlamaIndex **PropertyGraphIndex** to retrieve structured context and improve multi-hop reasoning.

---

## Author

**Tirumala Teja Yegineni**  
GitHub: https://github.com/TIRUMALA9999
