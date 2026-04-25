import os
from typing import Any, Dict, List, Optional

from pinecone import Pinecone, ServerlessSpec

from src.agent_architecture import CandidateRecord, PineconeConfig


class PineconeMatchingEngine:
    def __init__(
        self,
        api_key: Optional[str] = None,
        environment: Optional[str] = None,
        config: Optional[PineconeConfig] = None,
    ):
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        self.environment = environment or os.getenv("PINECONE_ENVIRONMENT")
        self.config = config or PineconeConfig()
        self.config.index_name = os.getenv(
            "PINECONE_INDEX", self.config.index_name)
        self.config.pod_type = os.getenv(
            "PINECONE_POD_TYPE", self.config.pod_type)

        if not self.api_key:
            raise ValueError("PINECONE_API_KEY is required")
        if not self.environment:
            raise ValueError("PINECONE_ENVIRONMENT is required")

        self.client = Pinecone(api_key=self.api_key,
                               environment=self.environment)
        self._ensure_index()
        self.index = self.client.Index(self.config.index_name)

    def _ensure_index(self) -> None:
        existing_indexes = self.client.list_indexes().names()
        if self.config.index_name not in existing_indexes:
            self.client.create_index(
                name=self.config.index_name,
                dimension=self.config.dimension,
                metric=self.config.metric,
                spec=ServerlessSpec(
                    cloud="aws",
                    region=self.environment,
                ),
            )

    def _metadata_for_candidate(self, candidate: CandidateRecord) -> Dict[str, Any]:
        return {
            "candidate_id": candidate.candidate_id,
            "name": candidate.name,
            "role": candidate.role,
            "experience_years": candidate.experience_years,
            "top_skills": candidate.top_skills,
            "location": candidate.location,
            "source": candidate.source,
            "current_state": candidate.current_state.value if hasattr(candidate.current_state, "value") else candidate.current_state,
            "match_score": candidate.match_score,
            "interest_score": candidate.interest_score,
            **candidate.metadata,
        }

    def upsert_candidates(
        self,
        candidates: List[CandidateRecord],
        embeddings: List[List[float]],
    ) -> Dict[str, Any]:
        if len(candidates) != len(embeddings):
            raise ValueError(
                "Candidates and embeddings must have the same length")

        vectors = []
        for candidate, vector in zip(candidates, embeddings):
            vectors.append(
                {
                    "id": candidate.candidate_id,
                    "values": vector,
                    "metadata": self._metadata_for_candidate(candidate),
                }
            )

        return self.index.upsert(vectors=vectors)

    def query(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self.index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            include_values=False,
            filter=filter,
        )


if __name__ == "__main__":
    engine = PineconeMatchingEngine(
        api_key=os.getenv("PINECONE_API_KEY"),
        environment=os.getenv("PINECONE_ENVIRONMENT"),
    )
    print(f"Connected to Pinecone index: {engine.config.index_name}")
