import os
import time
import streamlit as st

from dotenv import load_dotenv

from RAG_CHAIN.document_processor import DocumentProcessor
from RAG_CHAIN.retriever import DenseRetriever, SparseRetriever, HybridRetriever
from RAG_CHAIN.llm import build_llm, RAG_PROMPT

load_dotenv()

st.set_page_config(page_title="ResearchChat-AI", page_icon="📄")
st.title("ResearchChat-AI")
st.caption("Hybrid RAG (Dense + Sparse) — Chat with your research papers")


# ---------------------------------------------------------------------
# HybridRAGChain — ties retrieval + LLM together (previously rag_chain.py)
# ---------------------------------------------------------------------
class HybridRAGChain:
    def __init__(self):
        self.processor = DocumentProcessor()
        self.llm = build_llm()

        self.dense_retriever: DenseRetriever | None = None
        self.sparse_retriever: SparseRetriever | None = None
        self.hybrid_retriever: HybridRetriever | None = None
        self.documents: list | None = None

    @property
    def is_ready(self) -> bool:
        return self.hybrid_retriever is not None

    def build_index(self, directory_path: str) -> int:
        """
        Runs the full document -> chunks -> embeddings -> index pipeline.
        Returns the number of vectors indexed.
        """
        final_documents, document_embeddings = self.processor.process_directory(
            directory_path
        )

        self.documents = final_documents
        self.dense_retriever = DenseRetriever(document_embeddings, final_documents)
        self.sparse_retriever = SparseRetriever(final_documents)
        self.hybrid_retriever = HybridRetriever(
            self.dense_retriever, self.sparse_retriever
        )

        return self.dense_retriever.ntotal

    def query(self, user_prompt: str, k: int = 4) -> dict:
        """
        Runs retrieval + LLM generation for a single query.
        Returns a dict with the answer, retrieved docs, and elapsed time.
        """
        if not self.is_ready:
            raise RuntimeError("Index has not been built yet. Call build_index() first.")

        start_time = time.time()

        query_embedding = self.processor.embed_query(user_prompt)

        retrieved_documents = self.hybrid_retriever.search(
            query=user_prompt,
            query_embedding=query_embedding,
            k=k,
        )

        context = "\n\n".join(doc.page_content for doc in retrieved_documents)

        response = self.llm.invoke(
            RAG_PROMPT.format(context=context, input=user_prompt)
        )

        elapsed_time = time.time() - start_time

        return {
            "answer": response.content,
            "retrieved_documents": retrieved_documents,
            "elapsed_time": elapsed_time,
        }


# ---------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------

# ---- Initialize the RAG chain once per session ----
if "rag_chain" not in st.session_state:
    try:
        st.session_state.rag_chain = HybridRAGChain()
    except EnvironmentError as e:
        st.error(str(e))
        st.stop()


directory_path = os.path.join(os.getcwd(), "research_papers")


# ---- File upload ----
uploaded_files = st.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True,
)

if uploaded_files:
    os.makedirs(directory_path, exist_ok=True)

    for uploaded_file in uploaded_files:
        file_path = os.path.join(directory_path, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    st.success(f"Uploaded {len(uploaded_files)} file(s) successfully!")


# ---- Build index ----
if st.button("Document Embedding"):
    with st.spinner("Building hybrid index (FAISS + BM25)..."):
        try:
            n_vectors = st.session_state.rag_chain.build_index(directory_path)
            st.success(f"Vector database initialized successfully. {n_vectors} vectors indexed.")
        except (FileNotFoundError, ValueError) as e:
            st.error(str(e))


# ---- Query ----
user_prompt = st.text_input("Enter your query from the research paper")

if user_prompt:
    if st.session_state.rag_chain.is_ready:
        result = st.session_state.rag_chain.query(user_prompt, k=4)

        st.write(f"Response time: {result['elapsed_time']:.2f} seconds")
        st.write(result["answer"])

        with st.expander("Document similarity search"):
            for i, doc in enumerate(result["retrieved_documents"]):
                st.write(f"### Document {i + 1}")
                st.write(doc.page_content)
                st.write("-------------------")
    else:
        st.warning(
            "Vector database is not initialized. Please click 'Document Embedding' first."
        )