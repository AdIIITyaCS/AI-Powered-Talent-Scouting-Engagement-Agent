from typing import List
import os
from sentence_transformers import SentenceTransformer


class EmbeddingService:
    def __init__(self):
        # Local model load hoga (384 dimensions)
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        try:
            # Model locally run ho raha hai
            embeddings = self.model.encode(
                documents, convert_to_tensor=False
            ).tolist()
            return embeddings
        except Exception as exc:
            raise RuntimeError(f"Failed to create embeddings locally: {exc}")

    def embed_text(self, text: str) -> List[float]:
        try:
            embedding = self.model.encode(
                text, convert_to_tensor=False).tolist()
            return embedding
        except Exception as exc:
            raise RuntimeError(f"Failed to create embedding locally: {exc}")
