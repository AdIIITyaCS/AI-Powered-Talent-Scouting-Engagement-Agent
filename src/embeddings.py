import os
import requests
from typing import List


class EmbeddingService:
    def __init__(self, model_id: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Inference API version to save RAM on Render (Free Tier).
        Requires HF_TOKEN in environment variables.
        """
        self.api_url = f"https://api-inference.huggingface.co/models/{model_id}"
        self.headers = {"Authorization": f"Bearer {os.getenv('HF_TOKEN')}"}

    def embed_text(self, text: str) -> List[float]:
        """Get embedding for a single string (like JD)."""
        payload = {"inputs": text, "options": {"wait_for_model": True}}
        try:
            response = requests.post(
                self.api_url, headers=self.headers, json=payload)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Embedding Error: {e}")
            # Returns 384-dim zero vector as fallback
            return [0.0] * 384

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for a list of strings (Candidates).
        Sends all texts in one batch for better performance.
        """
        if not texts:
            return []

        payload = {"inputs": texts, "options": {"wait_for_model": True}}
        try:
            response = requests.post(
                self.api_url, headers=self.headers, json=payload)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Batch Embedding Error: {e}")
            return [[0.0] * 384 for _ in texts]
