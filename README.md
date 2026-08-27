# 🔬 ResearchChat-AI

### AI-Powered Research Paper Question Answering using RAG

**ResearchChat-AI** is a Retrieval-Augmented Generation (RAG) application that allows users to upload research papers in PDF format and ask questions about their content.

The application processes uploaded documents using **Docling**, splits them into meaningful chunks, converts the chunks into vector embeddings using **BAAI/bge-small-en**, stores the vectors in a **native FAISS index**, retrieves the most relevant document chunks for a user query, and generates a grounded answer using **Groq's GPT-OSS 20B model**.

---

## 🚀 Project Overview

Reading and understanding multiple research papers can be time-consuming. ResearchChat-AI provides a conversational interface where users can upload research papers and ask questions directly about their content.

Instead of sending the entire document to the LLM, the application uses **Retrieval-Augmented Generation** to retrieve only the most relevant sections before generating an answer.

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
🔢 Vector Embeddings
        ↓
⚡ Native FAISS
        ↓
🔎 Similarity Search
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
- 🔍 Semantic similarity search
- 🧩 Recursive document chunking
- 🧠 Hugging Face BGE embeddings
- ⚡ Native FAISS vector search
- 🤖 Groq-hosted GPT-OSS 20B
- 📚 Top-K relevant document retrieval
- 🎯 Context-grounded responses
- ⏱️ Response-time measurement
- 📖 View retrieved document chunks
- 🖥️ Interactive Streamlit interface

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| **Python** | Core programming language |
| **Streamlit** | Web application interface |
| **Docling** | PDF/document processing |
| **LangChain** | LLM and embedding integrations |
| **Hugging Face** | Embedding integration |
| **BAAI/bge-small-en** | Text embedding model |
| **FAISS** | Vector similarity search |
| **NumPy** | Numerical/vector processing |
| **Groq** | LLM inference |
| **GPT-OSS 20B** | Answer generation |
| **PyTorch** | Embedding model backend |

---

## 🧠 RAG Architecture

ResearchChat-AI follows a standard RAG architecture with a custom/native FAISS implementation.

### 1. Document Ingestion

Users upload PDF research papers through the Streamlit interface.

```text
PDF
 ↓
DoclingLoader
 ↓
Document Objects
```

### 2. Text Chunking

Documents are split using `RecursiveCharacterTextSplitter`.

Current configuration:

```python
chunk_size=1500
chunk_overlap=300
```

This produces smaller chunks that can be efficiently embedded and retrieved.

### 3. Embedding Generation

Each document chunk is converted into a numerical vector using:

```text
BAAI/bge-small-en
```

The embedding model is accessed through:

```python
HuggingFaceEmbeddings
```

### 4. Vector Storage

Instead of using LangChain's community FAISS wrapper, the project uses **FAISS directly**.

```python
index = faiss.IndexFlatL2(dimension)

index.add(document_embeddings)
```

This provides direct control over the vector index and similarity search process.

### 5. Query Retrieval

When a user asks a question:

```text
User Question
     ↓
Query Embedding
     ↓
FAISS Search
     ↓
Top 4 Similar Vectors
     ↓
Relevant Documents
```

The application currently retrieves the top **4** relevant document chunks.

### 6. Context Augmentation

The retrieved chunks are combined into the context provided to the LLM.

```text
Retrieved Documents
        +
User Question
        ↓
      Prompt
```

### 7. Answer Generation

The final prompt is sent to:

```text
Groq
  ↓
openai/gpt-oss-20b
```

The model is instructed to answer based only on the retrieved context.

---

## 🔎 Why Native FAISS?

A key implementation detail of this project is that FAISS is used directly rather than through:

```python
from langchain_community.vectorstores import FAISS
```

The project explicitly performs:

```python
index = faiss.IndexFlatL2(dimension)

index.add(document_embeddings)

distances, indices = index.search(
    query_embedding,
    k
)
```

This makes the underlying vector-search workflow transparent:

```text
Embedding Vector
      ↓
FAISS Index
      ↓
Similarity Search
      ↓
Vector Index
      ↓
Original Document
```

The FAISS index and document list are maintained separately:

```python
st.session_state.vectors
st.session_state.vector_documents
```

This approach helped me understand the underlying mechanics of vector stores and retrieval rather than treating the vector database as a black box.

---

## 📂 Project Structure

```text
ResearchChat-AI/
│
├── app1.py
├── requirements.txt
├── README.md
├── .env
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
```

```bash
cd ResearchChat-AI
```

### 2. Create a virtual environment

Windows:

```powershell
python -m venv myenv
```

Activate the environment:

```powershell
myenv\Scripts\activate
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file in the project root:

```text
GROQ_API_KEY=your_groq_api_key
```

### ⚠️ Important

Never commit your `.env` file or API keys to GitHub.

Add the following to `.gitignore`:

```text
.env
myenv/
__pycache__/
*.pyc
```

---

## ▶️ Run the Application

Start the Streamlit application:

```powershell
streamlit run app1.py
```

Then open the URL displayed by Streamlit in your browser.

---

## 💻 How to Use

### Step 1 — Upload Research Papers

Upload one or more PDF research papers.

### Step 2 — Create Vector Database

Click:

```text
Document Embedding
```

The application will:

```text
PDF
 ↓
Docling
 ↓
Document Chunks
 ↓
Embeddings
 ↓
FAISS Index
```

### Step 3 — Ask Questions

Enter a question related to the uploaded research papers.

Example:

```text
What is the main methodology used in this research paper?
```

The application retrieves the most relevant chunks and provides them to the LLM as context.

### Step 4 — Inspect Retrieved Context

Expand:

```text
Document similarity search
```

to see the chunks retrieved from FAISS.

---

## 📊 Current Configuration

| Component | Configuration |
|---|---|
| Document Processor | Docling |
| Chunk Size | 1500 |
| Chunk Overlap | 300 |
| Embedding Model | `BAAI/bge-small-en` |
| Vector Database | FAISS |
| FAISS Index | `IndexFlatL2` |
| Retrieval Count | Top 4 |
| LLM Provider | Groq |
| LLM | `openai/gpt-oss-20b` |
| UI | Streamlit |

---

## 📌 Key Learning Outcomes

This project demonstrates practical implementation of:

- Retrieval-Augmented Generation (RAG)
- Document ingestion
- PDF processing
- Text chunking
- Embedding generation
- Vector databases
- FAISS similarity search
- Semantic retrieval
- Prompt construction
- LLM-based question answering
- LangChain integrations
- Streamlit application development

---

## 🔮 Future Improvements

Planned improvements include:

- [ ] Persistent FAISS index storage
- [ ] Metadata-based filtering
- [ ] Configurable Top-K retrieval
- [ ] Hybrid search
- [ ] Re-ranking
- [ ] RAG evaluation using RAGAS
- [ ] Source/citation tracking
- [ ] Conversation memory
- [ ] Streaming responses
- [ ] Chat history
- [ ] Retrieval monitoring and observability
- [ ] Production deployment

---

## 🎯 Project Goal

The primary goal of ResearchChat-AI is to demonstrate how modern RAG systems can combine:

```text
Document Processing
        +
Embeddings
        +
Vector Search
        +
LLMs
        =
Context-Aware AI Applications
```

The project focuses on understanding the **end-to-end RAG pipeline**, including the underlying vector retrieval process.

---

## 👨‍💻 Author

**Kesava**

This project was developed as part of my learning and practical exploration of **Generative AI, RAG, LangChain, embeddings, vector databases, and LLM application development**.