# 🔬 ResearchChat-AI

### AI-Powered Research Paper Question Answering using Hybrid RAG

**ResearchChat-AI** is a Retrieval-Augmented Generation (RAG) application that lets users upload research papers in PDF format and ask questions about their content.

Uploaded documents are processed with **Docling**, split into chunks, embedded using **BAAI/bge-small-en**, and indexed with **native FAISS** (dense) and **BM25** (sparse). Both retrieval results are fused using **Reciprocal Rank Fusion (RRF)**, and a grounded answer is generated using **Groq's GPT-OSS 20B** model.

---

## 🚀 Project Overview

Reading and understanding multiple research papers is time-consuming. ResearchChat-AI provides a conversational interface where users can upload papers and ask questions directly about their content.

Instead of sending the entire document to the LLM, the application uses **hybrid retrieval** to fetch only the most relevant sections before generating an answer. Pure vector search can miss exact keyword or technical-term matches (acronyms, model names, equations), while pure keyword search misses semantic paraphrases. Combining both closes that gap.

### Core Pipeline

```text
📄 Research Papers
        ↓
   DoclingLoader
        ↓
📑 Documents
        ↓
Text Splitting
        ↓
🧩 Document Chunks
        ↓
Hugging Face Embeddings
        ↓
   ┌────────────┴────────────┐
   ↓                         ↓
🔢 Vector Embeddings     🔤 Tokenized Chunks
   ↓                         ↓
⚡ FAISS (Dense)          📇 BM25 (Sparse)
   ↓                         ↓
   └────────────┬────────────┘
                ↓
      🔀 Reciprocal Rank Fusion
                ↓
        📚 Top-K Relevant Chunks
                ↓
        📝 Context + Question
                ↓
        🤖 Groq GPT-OSS 20B
                ↓
        💬 Grounded Answer
```

---

## ✨ Features

- 📄 Upload multiple research paper PDFs
- 🔍 Hybrid semantic + keyword search
- 🧩 Recursive document chunking
- 🧠 Hugging Face BGE embeddings
- ⚡ Native FAISS vector search
- 🤖 Groq-hosted GPT-OSS 20B
- 📚 Top-K relevant chunk retrieval
- 🎯 Context-grounded responses
- ⏱️ Response-time measurement
- 📖 View retrieved document chunks
- 🖥️ Interactive Streamlit interface
- 🧱 Modular, package-based code structure

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| **Python** | Core programming language |
| **Streamlit** | Web application interface |
| **Docling** | PDF / document processing |
| **LangChain** | LLM and embedding integrations |
| **Hugging Face** | Embedding integration |
| **BAAI/bge-small-en** | Text embedding model |
| **FAISS** | Dense vector similarity search |
| **rank_bm25** | Sparse keyword-based search |
| **NumPy** | Numerical / vector processing |
| **Groq** | LLM inference |
| **GPT-OSS 20B** | Answer generation |
| **PyTorch** | Embedding model backend |

---

## 🧠 RAG Architecture

ResearchChat-AI follows a **hybrid RAG architecture**, combining a native FAISS implementation with BM25 keyword search, fused via RRF.

### 1. Document Ingestion

Users upload PDF research papers through the Streamlit interface.

```text
PDF → DoclingLoader → Document Objects
```

### 2. Text Chunking

Documents are split using `RecursiveCharacterTextSplitter`.

```python
chunk_size=1500      # characters
chunk_overlap=300    # characters
```

> Note: LangChain's `RecursiveCharacterTextSplitter` measures length in **characters** by default, so chunks are 1500 characters with a 300-character overlap.

### 3. Embedding Generation

Each chunk is converted into a vector using `BAAI/bge-small-en`, accessed through `HuggingFaceEmbeddings`.

### 4. Dual Indexing: Dense + Sparse

The project builds **two independent indexes over the same chunks**.

**Dense (FAISS)**

```python
index = faiss.IndexFlatL2(dimension)
index.add(document_embeddings)
```

**Sparse (BM25)**

```python
tokenized_corpus = [doc.page_content.lower().split() for doc in documents]
bm25 = BM25Okapi(tokenized_corpus)
```

### 5. Hybrid Retrieval: Reciprocal Rank Fusion (RRF)

When a user asks a question, both indexes are queried independently and their ranked lists are merged using RRF:

```text
User Question
      ↓
┌─────┴─────┐
↓           ↓
FAISS      BM25
Search     Search
↓           ↓
Dense      Sparse
Ranks      Ranks
└─────┬─────┘
      ↓
Reciprocal Rank Fusion
      ↓
Top-4 Fused Chunks
```

Each chunk is scored as:

```text
score(doc) = Σ 1 / (rrf_k + rank_in_retriever)
```

summed across the retrievers that returned it. This avoids normalizing FAISS L2 distances and BM25 scores onto a shared scale, and a chunk that ranks highly in *either* list surfaces near the top.

### 6. Context Augmentation

The fused top chunks are combined with the user question into a single prompt.

### 7. Answer Generation

The prompt is sent to Groq (`openai/gpt-oss-20b`), which is instructed to answer based only on the retrieved context.

---

## 🔎 Why Hybrid (FAISS + BM25) Instead of Vector Search Alone?

Retrieval is **not** limited to a single dense vector store. The project explicitly maintains two retrieval paths and fuses them:

```python
# Dense
index = faiss.IndexFlatL2(dimension)
index.add(document_embeddings)
distances, indices = index.search(query_embedding, k)

# Sparse
bm25 = BM25Okapi(tokenized_corpus)
scores = bm25.get_scores(tokenized_query)

# Fusion
rrf_scores[doc_idx] += 1 / (rrf_k + rank)
```

The FAISS index, BM25 index, and document list are kept as separate, composable objects rather than a single opaque vector store. This makes it possible to reason about *why* a chunk was retrieved and to swap or extend either strategy independently.

---

## 📊 Evaluation

> ⚠️ Fill this table with your **own measured results**. Do not publish numbers you have not measured.

**Setup:** [N] questions written from [X] research papers. A question counts as a hit if the chunk containing the answer appears in the top 4 results.

| Retrieval Method | Hit Rate @ 4 |
|---|---|
| Dense only (FAISS) | [X]% |
| Sparse only (BM25) | [X]% |
| Hybrid (FAISS + BM25 + RRF) | [X]% |

---

## 📂 Project Structure

```text
ResearchChat-AI/
│
├── RAG_CHAIN/                   # core RAG logic
│   ├── document_processor.py    # Docling load → chunk → BGE embed
│   ├── retriever.py             # FAISS (dense) + BM25 (sparse) + RRF (hybrid)
│   └── llm.py                   # prompt template + Groq LLM
│
├── app.py                       # Streamlit UI + RAG orchestration
├── requirements.txt
├── README.md
├── .env                         # not committed
├── .gitignore
│
└── research_papers/
    └── uploaded PDF files
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/ResearchChat-AI.git
cd ResearchChat-AI
```

### 2. Create a virtual environment

Windows:

```powershell
python -m venv myenv
myenv\Scripts\activate
```

macOS / Linux:

```bash
python3 -m venv myenv
source myenv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file in the project root:

```text
GROQ_API_KEY=your_groq_api_key
```

### ⚠️ Important

Never commit your `.env` file or API keys to GitHub. Add this to `.gitignore`:

```text
.env
myenv/
__pycache__/
*.pyc
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

Then open the URL shown in the terminal.

---

## 💻 How to Use

1. **Upload research papers:** upload one or more PDF files.
2. **Create the vector database:** click **Document Embedding**. The app parses the PDFs with Docling, chunks the text, generates embeddings, and builds the FAISS and BM25 indexes.
3. **Ask questions:** for example, *"What is the main methodology used in this research paper?"* The app runs dense and sparse retrieval, fuses results with RRF, and passes the top chunks to the LLM.
4. **Inspect retrieved context:** expand **Document similarity search** to see the chunks retrieved after hybrid fusion.

---

## 📊 Current Configuration

| Component | Configuration |
|---|---|
| Document Processor | Docling |
| Chunk Size | 1500 characters |
| Chunk Overlap | 300 characters |
| Embedding Model | `BAAI/bge-small-en` |
| Dense Index | FAISS `IndexFlatL2` |
| Sparse Index | BM25 (`rank_bm25`) |
| Fusion Strategy | Reciprocal Rank Fusion (RRF), `rrf_k=60` |
| Retrieval Count | Top 4 |
| LLM Provider | Groq |
| LLM | `openai/gpt-oss-20b` |
| UI | Streamlit |

---

## 📌 Key Learning Outcomes

- Retrieval-Augmented Generation (RAG)
- Document ingestion and PDF processing
- Text chunking and embedding generation
- Dense vector search (FAISS) and sparse keyword search (BM25)
- Hybrid retrieval and rank fusion (RRF)
- Prompt construction and LLM-based question answering
- LangChain integrations
- Modular Python package design
- Streamlit application development

---

## 🔮 Future Improvements

- [ ] Persistent FAISS + BM25 index storage
- [ ] Source citations with filename and page number
- [ ] Configurable Top-K retrieval
- [ ] Tunable hybrid weighting (dense vs. sparse)
- [ ] Cross-encoder re-ranking
- [ ] RAG evaluation using RAGAS
- [ ] Conversation memory and chat history
- [ ] Streaming responses
- [ ] Retrieval monitoring and observability
- [ ] Production deployment

---

## 🎯 Project Goal

To demonstrate how a modern RAG system combines document processing, embeddings, hybrid retrieval (dense + sparse + fusion), and LLMs into a context-aware application, with a focus on understanding the end-to-end hybrid pipeline and how the results are fused.

---

## 👨‍💻 Author

**Kesava Pavan Gadde**
B.Tech in Computer Science (Artificial Intelligence), Parul University

- 📧 kesavapavangadde@gmail.com
- 💼 [LinkedIn](https://www.linkedin.com/in/kesavapavan-gadde-26a3b5263/)
- 🐙 [GitHub](https://github.com/<your-username>)

Developed as part of my hands-on learning in Generative AI, RAG, LangChain, embeddings, vector databases, and LLM application development.
