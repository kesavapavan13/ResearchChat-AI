import faiss
import numpy as np

from rank_bm25 import BM25Okapi


class DenseRetriever:
    """FAISS-based vector similarity retrieval."""

    def __init__(self, document_embeddings: np.ndarray, documents: list):
        self.documents = documents
        dimension = document_embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(document_embeddings)

    @property
    def ntotal(self) -> int:
        return self.index.ntotal

    def search(self, query_embedding: np.ndarray, k: int = 10):
        """Returns a list of (doc_index, distance) sorted by relevance (best first)."""
        distances, indices = self.index.search(query_embedding, k)

        results = []
        for rank, idx in enumerate(indices[0]):
            if idx == -1:
                continue
            results.append((int(idx), float(distances[0][rank])))

        return results


class SparseRetriever:
    """BM25-based keyword retrieval."""

    def __init__(self, documents: list):
        self.documents = documents
        tokenized_corpus = [
            doc.page_content.lower().split() for doc in documents
        ]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, k: int = 10):
        """Returns a list of (doc_index, score) sorted by relevance (best first)."""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        top_indices = np.argsort(scores)[::-1][:k]
        return [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]


class HybridRetriever:
    """
    Combines dense + sparse retrieval using Reciprocal Rank Fusion (RRF).

    RRF score for a document = sum over retrievers of 1 / (rrf_k + rank)
    where `rank` is the document's 1-indexed position in that retriever's
    result list. Documents that show up near the top of either list rank
    highly overall, without needing to normalize dense distances and BM25
    scores onto the same scale.
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: SparseRetriever,
        rrf_k: int = 60,
    ):
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.rrf_k = rrf_k

    def search(self, query: str, query_embedding: np.ndarray, k: int = 4, fetch_k: int = 20):
        dense_results = self.dense_retriever.search(query_embedding, k=fetch_k)
        sparse_results = self.sparse_retriever.search(query, k=fetch_k)

        rrf_scores: dict[int, float] = {}

        for rank, (doc_idx, _) in enumerate(dense_results, start=1):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + 1.0 / (self.rrf_k + rank)

        for rank, (doc_idx, _) in enumerate(sparse_results, start=1):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + 1.0 / (self.rrf_k + rank)

        ranked = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
        top_indices = [doc_idx for doc_idx, _ in ranked[:k]]

        documents = self.dense_retriever.documents
        return [documents[idx] for idx in top_indices]
