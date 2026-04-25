from src.scout_agent import PeopleDataLabsScout
from src.matching_engine import PineconeMatchingEngine
from src.jd_analyst import AffindaJDAnalyst
from src.engagement_bot import EngagementBot
from src.embeddings import EmbeddingService
from src.agent_architecture import CandidateRecord, CandidateState, FinalScoreCalculator, JDMetadata
from typing import List, Dict, Any
import argparse
import os
from dotenv import load_dotenv
load_dotenv()


class TalentScoutingOrchestrator:
    def __init__(self):
        self.jd_analyzer = AffindaJDAnalyst()
        self.scout = PeopleDataLabsScout()
        self.embeddings = EmbeddingService()
        self.matching_engine = PineconeMatchingEngine(
            api_key=os.getenv("PINECONE_API_KEY"),
            environment=os.getenv("PINECONE_ENVIRONMENT"),
        )
        self.engagement_bot = EngagementBot()
        self.score_calculator = FinalScoreCalculator()

    def run(
        self,
        jd_text: str,
        candidate_limit: int = 8,
        top_k: int = 5,
        debug: bool = False,
    ) -> List[Dict[str, Any]]:
        jd_metadata = self.jd_analyzer.parse_job_description(
            jd_text, debug=debug)
        return self._run_with_metadata(jd_metadata, candidate_limit, top_k, debug)

    def run_from_file(
        self,
        file_bytes: bytes,
        filename: str,
        candidate_limit: int = 8,
        top_k: int = 5,
        debug: bool = False,
    ) -> List[Dict[str, Any]]:
        jd_metadata = self.jd_analyzer.parse_job_description_from_bytes(
            file_bytes, filename, debug=debug)
        if debug:
            print("\n[Parsed JD Metadata from file]", jd_metadata)
        return self._run_with_metadata(jd_metadata, candidate_limit, top_k, debug)

    def _run_with_metadata(
        self,
        jd_metadata: JDMetadata,
        candidate_limit: int,
        top_k: int,
        debug: bool,
    ) -> List[Dict[str, Any]]:
        if debug:
            print("\n[Parsed JD Metadata]", jd_metadata)
        candidates = self.scout.discover_candidates(
            jd_metadata, limit=candidate_limit, debug=debug)

        if not candidates:
            raise RuntimeError("No candidates discovered for the given JD")

        candidate_texts = [
            f"{candidate.role} {' '.join(candidate.top_skills)} {candidate.location}"
            for candidate in candidates
        ]
        if debug:
            print("\n[Candidate texts to embed]:")
            for index, text in enumerate(candidate_texts[:5], start=1):
                print(f" {index}. {text}")
        candidate_embeddings = self.embeddings.embed_documents(candidate_texts)
        if debug and candidate_embeddings:
            print("\n[Embedding debug] vector length:",
                  len(candidate_embeddings[0]))
            print("[Embedding debug] first vector sample:",
                  candidate_embeddings[0][:8])
        self.matching_engine.upsert_candidates(
            candidates, candidate_embeddings)

        job_vector = self.embeddings.embed_text(
            f"{jd_metadata.title} {' '.join(jd_metadata.skills)} {jd_metadata.description}"
        )
        query_filter = {
            "current_state": {
                "$eq": CandidateState.DISCOVERED.value
            }
        }
        results = self.matching_engine.query(
            job_vector, top_k=top_k, filter=query_filter)

        if debug:
            print("\n[Matching results received]", results.get("matches", []))

        top_matches = []
        for match in results.get("matches", []):
            metadata = match.get("metadata", {})
            candidate = CandidateRecord(
                candidate_id=metadata.get("candidate_id", match["id"]),
                name=metadata.get("name", "Unknown"),
                role=metadata.get("role", "Unknown"),
                experience_years=int(metadata.get("experience_years", 0) or 0),
                top_skills=metadata.get("top_skills", []),
                location=metadata.get("location", "Unknown"),
                source=metadata.get("source", "PeopleDataLabs"),
                current_state=CandidateState.MATCHED,
                match_score=float(match.get("score", 0.0)),
                interest_score=0.0,
                metadata=metadata,
            )
            response_text = self.engagement_bot.simulate_outreach(
                candidate, jd_metadata.title)
            interest_score = self.engagement_bot.score_candidate(
                candidate, response_text)
            candidate.current_state = CandidateState.ENGAGED
            final_score = self.score_calculator.calculate(
                candidate.match_score, interest_score)
            if debug:
                print("\n[Engagement Debug]")
                print(f" candidate_id: {candidate.candidate_id}")
                print(f" response_text: {response_text}")
                print(f" interest_score: {interest_score}")
                print(f" final_score: {final_score}\n")
            top_matches.append({
                "candidate_id": candidate.candidate_id,
                "name": candidate.name,
                "role": candidate.role,
                "match_score": round(candidate.match_score, 4),
                "interest_score": round(candidate.interest_score, 4),
                "final_score": round(final_score, 4),
                "state": candidate.current_state.value,
                "response_text": response_text,
            })

        top_matches.sort(key=lambda item: item["final_score"], reverse=True)
        return top_matches


if __name__ == "__main__":
    default_text = """
    Job Title: Senior AI Agent Developer
    Location: Remote (US-based preferred)
    Experience: 5+ years

    About the Role:
    We are looking for a Senior AI Agent Developer to design and build 
    autonomous AI agents using LLMs, RAG pipelines, and multi-agent frameworks.

    Requirements:
    - 5+ years experience in Python, ML/AI engineering
    - Experience with LangChain, LlamaIndex, or similar frameworks
    - Strong background in NLP, embeddings, and vector databases
    - Experience with Pinecone, Weaviate, or similar
    - Familiarity with FastAPI, Docker, Kubernetes

    Nice to Have:
    - Experience with multi-agent orchestration
    - Knowledge of reinforcement learning
    - Background in recruitment technology

    Salary: $150,000 - $200,000
    Industry: Artificial Intelligence / SaaS
    """

    parser = argparse.ArgumentParser(
        description="Run the talent scouting orchestrator with a job description."
    )
    parser.add_argument(
        "--jd",
        type=str,
        default=default_text,
        help="Job description text to use for candidate discovery.",
    )
    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=8,
        help="Maximum number of candidates to discover from PDL.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top matches to return from Pinecone.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug information for the pipeline.",
    )
    args = parser.parse_args()

    orchestrator = TalentScoutingOrchestrator()
    results = orchestrator.run(
        args.jd,
        candidate_limit=args.candidate_limit,
        top_k=args.top_k,
        debug=args.debug,
    )
    for result in results:
        print(result)
