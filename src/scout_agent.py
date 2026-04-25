import os
from typing import List, Dict, Any, Optional
import json
import requests

from src.agent_architecture import CandidateRecord, CandidateState, JDMetadata


class PeopleDataLabsScout:
    """Discover candidates using People Data Labs with simple structured filters."""

    BASE_URL = "https://api.peopledatalabs.com/v5/person/search"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("PDL_API_KEY")
        if not self.api_key:
            raise ValueError(
                "People Data Labs API key is required via PDL_API_KEY")

        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "X-Api-Key": self.api_key,
        })

    def build_search_payload(self, jd: JDMetadata, limit: int = 50) -> Dict[str, Any]:
        effective_limit = min(limit, 50)
        query_text = jd.title if jd.title and jd.title.lower(
        ) != "unknown" else jd.description[:256]
        payload: Dict[str, Any] = {
            "limit": effective_limit,
            "query": json.dumps({
                "bool": {
                    "must": [
                        {"match": {"job_title": query_text}}
                    ]
                }
            }),
        }

        if jd.skills:
            payload["skills"] = [skill.lower() for skill in jd.skills[:50]]

        if jd.location and jd.location.lower() not in {"global", "unknown"}:
            normalized = jd.location.strip().lower()
            if "remote" in normalized:
                payload["location"] = "Remote"
            elif "united states" in normalized or "usa" in normalized or "us-based" in normalized or "u.s." in normalized:
                payload["location"] = "United States"
            else:
                payload["location"] = jd.location.strip()

        if jd.experience_years > 0:
            payload["experience_years"] = jd.experience_years

        if jd.industry:
            payload["industry"] = jd.industry

        if jd.additional_filters:
            payload.update(jd.additional_filters)

        return payload

    def _execute_search(self, payload: Dict[str, Any]) -> requests.Response:
        return self.session.post(self.BASE_URL, json=payload, timeout=30)

    def search_candidates(self, jd: JDMetadata, limit: int = 50, debug: bool = False) -> List[Dict[str, Any]]:
        payload = self.build_search_payload(jd, limit=limit)
        if debug:
            print("\n[PDL Search Payload]:")
            print(json.dumps(payload, indent=2, ensure_ascii=False))

        response = self._execute_search(payload)
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            raise RuntimeError(
                f"People Data Labs search failed: {response.status_code} - {response.text}\n"
                f"Payload: {payload}"
            ) from exc

        data = response.json()
        if debug:
            print("\n[PDL Response Data]:")
            print(json.dumps(data, indent=2, ensure_ascii=False)[:2000])
        return data.get("data", [])

    def normalize_candidate(
        self,
        raw_profile: Dict[str, Any],
        search_payload: Optional[Dict[str, Any]] = None,
    ) -> CandidateRecord:
        metadata: Dict[str, Any] = {
            "raw_profile": json.dumps(raw_profile)
        }
        if search_payload is not None:
            metadata["search_payload"] = json.dumps(search_payload)

        experience_years = raw_profile.get("inferred_years_experience")
        if experience_years is None:
            experience_years = raw_profile.get("experience_years", 0)

        return CandidateRecord(
            candidate_id=str(
                raw_profile.get("id", raw_profile.get("person_id", "unknown"))
            ),
            name=raw_profile.get("full_name", "Unknown"),
            role=raw_profile.get("job_title", "Unknown"),
            experience_years=int(experience_years or 0),
            top_skills=raw_profile.get("job_skills", []),
            location=raw_profile.get("location_name", "Unknown"),
            source="PeopleDataLabs",
            current_state=CandidateState.DISCOVERED,
            metadata=metadata,
        )

    def discover_candidates(self, jd: JDMetadata, limit: int = 50, debug: bool = False) -> List[CandidateRecord]:
        payload = self.build_search_payload(jd, limit=limit)
        raw_profiles = self.search_candidates(jd, limit=limit, debug=debug)
        if debug:
            print(f"\n[PDL returned {len(raw_profiles)} candidate profiles]")
        candidates = [
            self.normalize_candidate(profile, search_payload=payload)
            for profile in raw_profiles
        ]
        return candidates
