"""Functionality to retrieve relevant context from the database."""
import json
from typing import Optional, Any, List
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.core.schema import NodeWithScore
from llama_index.core import QueryBundle
from llama_index.core.retrievers import BaseRetriever
import psycopg2

from db import vector_store

class VectorDBRetriever(BaseRetriever):
    """Retriever over a postgres vector store"""

    def __init__(
        self,
        vector_store: PGVectorStore,
        embed_model: Any,
        query_mode: str = "default",
        similarity_top_k: int = 2,
    ) -> None:
        """Constructor for the retriever"""
        self._vector_store = vector_store
        self._embed_model = embed_model
        self._query_mode = query_mode
        self._similarity_top_k = similarity_top_k
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve"""

        # Query the DB
        query_embedding = embed_model.get_query_embedding(
            query_bundle.query_str
        )
        vector_store_query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=self._similarity_top_k,
            mode=self._query_mode,
        )
        query_result = vector_store.query(vector_store_query)

        # Return a set of candidate nodes
        nodes_with_scores = []
        for index, node in enumerate(query_result.nodes):
            score: Optional[float] = None
            if query_result.similarities is not None:
                score = query_result.similarities[index]
            nodes_with_scores.append(NodeWithScore(node=node, score=score))

        return nodes_with_scores


with open("./paths.json", "r") as f:
    paths = json.load(f)
    name = "BAAI/bge-small-en"
    if "bge-small-en" in paths:
        name = paths["bge-small-en"]

    # Embedding model
    embed_model = HuggingFaceEmbedding(model_name=name)

# Retriever
loader = VectorDBRetriever(
    vector_store, embed_model, query_mode="default", similarity_top_k=2
)
