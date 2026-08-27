**# 🔬 ResearchChat-AI**

**### AI-Powered Research Paper Question Answering using Hybrid RAG**

**\*\*ResearchChat-AI\*\*** is a Retrieval-Augmented Generation (RAG) application that allows users to upload research papers in PDF format and ask questions about their content.

The application processes uploaded documents using **\*\*Docling\*\***, splits them into meaningful chunks, converts the chunks into vector embeddings using **\*\*BAAI/bge-small-en\*\***, indexes them with **\*\*native FAISS\*\*** (dense) and **\*\*BM25\*\*** (sparse), fuses both retrieval strategies using **\*\*Reciprocal Rank Fusion (RRF)\*\***, and generates a grounded answer using **\*\*Groq's GPT-OSS 20B model\*\***.

---

**## 🚀 Project Overview**

Reading and understanding multiple research papers can be time-consuming. ResearchChat-AI provides a conversational interface where users can upload research papers and ask questions directly about their content.

Instead of sending the entire document to the LLM, the application uses **\*\*Retrieval-Augmented Generation\*\*** — and specifically **\*\*hybrid retrieval\*\*** — to retrieve only the most relevant sections before generating an answer. Pure vector search alone can miss exact keyword/technical-term matches (e.g. acronyms, model names, equations), while pure keyword search misses semantic paraphrases. Combining both closes that gap.

**### Core Pipeline**

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

   ↓                         ↓

🔢 Vector Embeddings     🔤 Tokenized Chunks

   ↓                         ↓

⚡ FAISS (Dense)          📇 BM25 (Sparse)

   ↓                         ↓

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

**## ✨ Features**

- 📄 Upload multiple research paper PDFs

- 🔍 Hybrid semantic + keyword search

- 🧩 Recursive document chunking

- 🧠 Hugging Face BGE embeddings

- ⚡ Native FAISS vector search

- 🤖 Groq-hosted GPT-OSS 20B

- 📚 Top-K relevant document retrieval

- 🎯 Context-grounded responses

- ⏱️ Response-time measurement

- 📖 View retrieved document chunks

- 🖥️ Interactive Streamlit interface

- 🧱 Modular, package-based code structure

---

**## 🛠️ Tech Stack**

\| Technology | Purpose |

\|---|---|

\| **\*\*Python\*\*** | Core programming language |

\| **\*\*Streamlit\*\*** | Web application interface |

\| **\*\*Docling\*\*** | PDF/document processing |

\| **\*\*LangChain\*\*** | LLM and embedding integrations |

\| **\*\*Hugging Face\*\*** | Embedding integration |

\| **\*\*BAAI/bge-small-en\*\*** | Text embedding model |

\| **\*\*FAISS\*\*** | Dense vector similarity search |

\| **\*\*rank_bm25\*\*** | Sparse keyword-based search |

\| **\*\*NumPy\*\*** | Numerical/vector processing |

\| **\*\*Groq\*\*** | LLM inference |

\| **\*\*GPT-OSS 20B\*\*** | Answer generation |

\| **\*\*PyTorch\*\*** | Embedding model backend |

---

**## 🧠 RAG Architecture**

ResearchChat-AI follows a **\*\*hybrid RAG architecture\*\***, combining a custom/native FAISS implementation with BM25 keyword search, fused via RRF.

**### 1. Document Ingestion**

Users upload PDF research papers through the Streamlit interface.

```text

PDF

 ↓

DoclingLoader

 ↓

Document Objects

```

**### 2. Text Chunking**

Documents are split using `RecursiveCharacterTextSplitter`.

Current configuration:

```python

chunk_size=1500

chunk_overlap=300

```

This produces smaller chunks that can be efficiently embedded, indexed, and retrieved.

**### 3. Embedding Generation**

Each document chunk is converted into a numerical vector using:

```text

BAAI/bge-small-en

```

The embedding model is accessed through:

```python

HuggingFaceEmbeddings

```

**### 4. Dual Indexing — Dense + Sparse**

Instead of relying on a single retrieval strategy, the project builds **\*\*two independent indexes over the same chunks\*\***:

**\*\*Dense — FAISS\*\***

```python

index = faiss.IndexFlatL2(dimension)

index.add(document_embeddings)

```

**\*\*Sparse — BM25\*\***

```python

tokenized_corpus = [doc.page_content.lower().split() for doc in documents]

bm25 = BM25Okapi(tokenized_corpus)

```

**### 5. Hybrid Retrieval — Reciprocal Rank Fusion (RRF)**

When a user asks a question, both indexes are queried independently, and their ranked result lists are merged using RRF:

```text

User Question

       ↓

 ┌─────┴─────┐

 ↓           ↓

FAISS      BM25

Search    Search

 ↓           ↓

Dense      Sparse

Ranks      Ranks

 ↓           ↓

 └─────┬─────┘

       ↓

Reciprocal Rank Fusion

       ↓

Top-4 Fused Chunks

```

RRF scores each document as:

```text

score(doc) = Σ  1 / (rrf_k + rank_in_retriever)

```

summed across whichever retrievers returned it. This avoids having to normalize FAISS L2 distances and BM25 scores onto a shared scale — a chunk that ranks highly in *\*either\** list surfaces near the top.

**### 6. Context Augmentation**

The retrieved chunks are combined into the context provided to the LLM.

```text

Fused Top-K Documents

          +

    User Question

          ↓

        Prompt

```

**### 7. Answer Generation**

The final prompt is sent to:

```text

Groq

  ↓

openai/gpt-oss-20b

```

The model is instructed to answer based only on the retrieved context.

---

**## 🔎 Why Hybrid (FAISS + BM25) Instead of Vector Search Alone?**

A key implementation detail of this project is that retrieval is **\*\*not\*\*** limited to a single dense vector store:

```python

from langchain_community.vectorstores import FAISS   # not used alone

```

Instead, the project explicitly maintains two independent retrieval paths and fuses them:

```python

\# Dense

index = faiss.IndexFlatL2(dimension)

index.add(document_embeddings)

distances, indices = index.search(query_embedding, k)

\# Sparse

bm25 = BM25Okapi(tokenized_corpus)

scores = bm25.get_scores(tokenized_query)

\# Fusion

rrf_scores[doc_idx] += 1 / (rrf_k + rank)

```

This makes the underlying retrieval workflow transparent end-to-end:

```text

Query

  ↓

Dense Search (semantic)  +  Sparse Search (lexical)

  ↓

Rank Fusion (RRF)

  ↓

Original Document Chunks

```

The FAISS index, BM25 index, and document list are maintained as separate, composable objects rather than a single opaque vector store — which made it possible to reason about *\*why\** a particular chunk was retrieved, and to swap or extend either retrieval strategy independently.

---

**## 📂 Project Structure**

```text

ResearchChat-AI/

│

├── myenv/                       # virtual environment (not committed)

├── RAG_CHAIN/                   # core RAG logic

│   ├── document_processor.py    # Docling load → chunk → BGE embed

│   ├── retriever.py             # FAISS (dense) + BM25 (sparse) + RRF (hybrid)

│   └── llm.py                    # prompt template + Groq LLM

│

├── app.py                        # Streamlit UI + RAG orchestration

├── requirements.txt

├── README.md

├── .env

├── .gitignore

│

└── research_papers/

    └── uploaded PDF files

```

---

**## ⚙️ Installation**

**### 1. Clone the repository**

```bash

git clone https://github.com/\<your-username>/ResearchChat-AI.git

```

```bash

cd ResearchChat-AI

```

**### 2. Create a virtual environment**

Windows:

```powershell

python -m venv myenv

```

Activate the environment:

```powershell

myenv\Scripts\activate

```

**### 3. Install dependencies**

```powershell

python -m pip install -r requirements.txt

```

---

**## 🔑 Environment Variables**

Create a `.env` file in the project root:

```text

GROQ_API_KEY=your_groq_api_key

```

**### ⚠️ Important**

Never commit your `.env` file or API keys to GitHub.

Add the following to `.gitignore`:

```text

.env

myenv/

__pycache__/

\*.pyc

```

---

**## ▶️ Run the Application**

Start the Streamlit application:

```powershell

streamlit run app.py

```

Then open the URL displayed by Streamlit in your browser.

---

**## 💻 How to Use**

**### Step 1 — Upload Research Papers**

Upload one or more PDF research papers.

**### Step 2 — Create Vector Database**

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

FAISS Index  +  BM25 Index

```

**### Step 3 — Ask Questions**

Enter a question related to the uploaded research papers.

Example:

```text

What is the main methodology used in this research paper?

```

The application runs both dense and sparse retrieval, fuses the results with RRF, and provides the top chunks to the LLM as context.

**### Step 4 — Inspect Retrieved Context**

Expand:

```text

Document similarity search

```

to see the chunks retrieved after hybrid fusion.

---

**## 📊 Current Configuration**

\| Component | Configuration |

\|---|---|

\| Document Processor | Docling |

\| Chunk Size | 1500 |

\| Chunk Overlap | 300 |

\| Embedding Model | `BAAI/bge-small-en` |

\| Dense Index | FAISS `IndexFlatL2` |

\| Sparse Index | BM25 (`rank_bm25`) |

\| Fusion Strategy | Reciprocal Rank Fusion (RRF), `rrf_k=60` |

\| Retrieval Count | Top 4 |

\| LLM Provider | Groq |

\| LLM | `openai/gpt-oss-20b` |

\| UI | Streamlit |

---

**## 📌 Key Learning Outcomes**

This project demonstrates practical implementation of:

- Retrieval-Augmented Generation (RAG)

- Document ingestion

- PDF processing

- Text chunking

- Embedding generation

- Dense vector search (FAISS)

- Sparse keyword search (BM25)

- Hybrid retrieval and rank fusion (RRF)

- Prompt construction

- LLM-based question answering

- LangChain integrations

- Modular Python package design

- Streamlit application development

---

**## 🔮 Future Improvements**

Planned improvements include:

- [ ] Persistent FAISS + BM25 index storage

- [ ] Metadata-based filtering (filename/page citation)

- [ ] Configurable Top-K retrieval

- [ ] Tunable hybrid weighting (dense vs. sparse)

- [ ] Cross-encoder re-ranking

- [ ] RAG evaluation using RAGAS

- [ ] Source/citation tracking

- [ ] Conversation memory

- [ ] Streaming responses

- [ ] Chat history

- [ ] Retrieval monitoring and observability

- [ ] Production deployment

---

**## 🎯 Project Goal**

The primary goal of ResearchChat-AI is to demonstrate how modern RAG systems can combine:

```text

Document Processing

        +

Embeddings

        +

Hybrid Retrieval (Dense + Sparse + Fusion)

        +

LLMs

        =

Context-Aware AI Applications

```

The project focuses on understanding the **\*\*end-to-end hybrid RAG pipeline\*\***, including the underlying dense and sparse retrieval mechanics and how they're fused.

---

**## 👨‍💻 Author**

**\*\*Kesava\*\***

This project was developed as part of my learning and practical exploration of **\*\*Generative AI, RAG, LangChain, embeddings, vector databases, and LLM application development\*\***.
