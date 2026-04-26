import re
from typing import Optional
from transformers import pipeline
from src.agent_architecture import CandidateRecord, FinalScoreCalculator


class EngagementBot:
    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        # Local pipeline load hogi
        self.sentiment = pipeline("sentiment-analysis", model=model_name)
        self.score_calculator = FinalScoreCalculator()

    def simulate_outreach(self, candidate: CandidateRecord, jd_title: str) -> str:
        skill_phrase = ', '.join(
            candidate.top_skills[:3]) if candidate.top_skills else 'your background'
        return (
            f"Hi {candidate.name}, thank you for reviewing this opportunity for {jd_title}. "
            f"Your experience with {skill_phrase} looks like a strong fit. "
            "Would you be interested in a brief conversation?"
        )

    def evaluate_response(self, response_text: str) -> float:
        if not response_text:
            return 0.0

        # Local processing
        sentiment_result = self.sentiment(response_text[:512])[0]
        base_score = 0.9 if sentiment_result["label"] == "POSITIVE" else 0.1
        confidence = float(sentiment_result["score"])
        interest_score = base_score * confidence

        strong_keywords = ["excited", "interested",
                           "keen", "love", "open", "available"]
        if any(re.search(rf"\b{keyword}\b", response_text, re.IGNORECASE) for keyword in strong_keywords):
            interest_score = min(1.0, interest_score + 0.15)

        if re.search(r"not interested|too busy|not looking|no thanks", response_text, re.IGNORECASE):
            interest_score = min(1.0, interest_score * 0.35)

        return round(float(interest_score), 3)

    def score_candidate(self, candidate: CandidateRecord, response_text: Optional[str] = None) -> float:
        if response_text is None:
            response_text = self.simulate_outreach(candidate, "")
        candidate.interest_score = self.evaluate_response(response_text)
        return candidate.interest_score
