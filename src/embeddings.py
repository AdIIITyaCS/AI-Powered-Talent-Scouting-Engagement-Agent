import os
import requests
from typing import List, Any


class EmbeddingService:
    def __init__(self, model_id: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Inference API version to save RAM on Render (Free Tier).
        Requires HF_TOKEN in environment variables.
        """
        self.api_url = f"https://api-inference.huggingface.co/models/{model_id}"
        token = os.getenv("HF_TOKEN")
        if not token:
            raise ValueError(
                "HF_TOKEN is required for Hugging Face embeddings")
        self.headers = {"Authorization": f"Bearer {token}"}

    def _parse_embedding_response(self, response: Any) -> List[float]:
        if isinstance(response, dict) and response.get("error"):
            raise ValueError(response["error"])

        if isinstance(response, list):
            if not response:
                raise ValueError("Empty embedding response")
            if all(isinstance(item, float) or isinstance(item, int) for item in response):
                return [float(value) for value in response]
            if all(isinstance(item, list) for item in response):
                raise ValueError(
                    "Batch response received for single text input")
            if all(isinstance(item, dict) for item in response) and "embedding" in response[0]:
                return [float(value) for value in response[0]["embedding"]]

        if isinstance(response, dict) and "embedding" in response:
            return [float(value) for value in response["embedding"]]

        raise ValueError("Unexpected embedding response format")

    def _validate_vector(self, vector: List[float]) -> bool:
        return bool(vector) and any(value != 0.0 for value in vector)

    def embed_text(self, text: str) -> List[float]:
        """Get embedding for a single string (like JD)."""
        payload = {"inputs": text, "options": {"wait_for_model": True}}
        response = requests.post(
            self.api_url, headers=self.headers, json=payload, timeout=60
        )
        response.raise_for_status()
        vector = self._parse_embedding_response(response.json())
        if not self._validate_vector(vector):
            raise ValueError("Embedding returned an all-zero vector")
        return vector

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for a list of strings (Candidates).
        Sends all texts in one batch for better performance.
        """
        if not texts:
            return []

        payload = {"inputs": texts, "options": {"wait_for_model": True}}
        response = requests.post(
            self.api_url, headers=self.headers, json=payload, timeout=120
        )
        response.raise_for_status()
        results = response.json()

        if not isinstance(results, list):
            raise ValueError("Unexpected batch embedding response format")

        embeddings = []
        for item in results:
            vector = self._parse_embedding_response(item)
            if not self._validate_vector(vector):
                raise ValueError("One or more embeddings returned all zeros")
            embeddings.append(vector)
        return embeddings
