import os
import numpy as np

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_docling.loader import DoclingLoader


class DocumentProcessor:
    def __init__(
        self,
        embedding_model: str = "BAAI/bge-small-en",
        chunk_size: int = 1500,
        chunk_overlap: int = 300,
        max_docs: int = 50,
    ):
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.max_docs = max_docs

    def list_pdf_files(self, directory_path: str) -> list[str]:
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory `{directory_path}` does not exist.")

        pdf_files = [
            os.path.join(directory_path, file)
            for file in os.listdir(directory_path)
            if file.lower().endswith(".pdf")
        ]

        if not pdf_files:
            raise ValueError("No PDF files found in the given directory.")

        return pdf_files

    def load_documents(self, pdf_files: list[str]) -> list:
        loader = DoclingLoader(file_path=pdf_files)
        docs = loader.load()

        if not docs:
            raise ValueError("No documents were loaded from the PDFs.")

        return docs

    def split_documents(self, docs: list) -> list:
        limited_docs = docs[: self.max_docs]
        final_documents = self.text_splitter.split_documents(limited_docs)

        if not final_documents:
            raise ValueError("Failed to split documents into chunks.")

        return final_documents

    def embed_documents(self, documents: list) -> np.ndarray:
        doc_texts = [doc.page_content for doc in documents]
        document_embeddings = self.embeddings.embed_documents(doc_texts)

        if not document_embeddings:
            raise ValueError("Embeddings were not generated.")

        return np.array(document_embeddings, dtype="float32")

    def embed_query(self, query: str) -> np.ndarray:
        query_embedding = self.embeddings.embed_query(query)
        return np.array([query_embedding], dtype="float32")

    def process_directory(self, directory_path: str):
        """
        Full pipeline: directory -> pdf_files -> docs -> chunks -> embeddings.
        Returns (final_documents, document_embeddings).
        """
        pdf_files = self.list_pdf_files(directory_path)
        docs = self.load_documents(pdf_files)
        final_documents = self.split_documents(docs)
        document_embeddings = self.embed_documents(final_documents)

        return final_documents, document_embeddings
