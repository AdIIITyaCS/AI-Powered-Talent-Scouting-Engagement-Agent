from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any


class CandidateState(Enum):
    DISCOVERED = "Discovered"
    MATCHED = "Matched"
    ENGAGED = "Engaged"
    SHORTLISTED = "Shortlisted"


@dataclass
class CandidateRecord:
    candidate_id: str
    name: str
    role: str
    experience_years: int
    top_skills: List[str]
    location: str
    source: str
    current_state: CandidateState = CandidateState.DISCOVERED
    match_score: float = 0.0
    interest_score: float = 0.0
    notes: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class JDMetadata:
    title: str
    skills: List[str]
    seniority: str
    location: str
    description: str
    industry: str = ""
    experience_years: int = 0
    additional_filters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PineconeConfig:
    index_name: str = "talent-matching-index"
    dimension: int = 384
    metric: str = "cosine"
    pod_type: str = "serverless"


class FinalScoreCalculator:
    def __init__(self, w_match: float = 0.7, w_interest: float = 0.3):
        self.w_match = w_match
        self.w_interest = w_interest

    def calculate(self, match_score: float, interest_score: float) -> float:
        return (self.w_match * match_score) + (self.w_interest * interest_score)
