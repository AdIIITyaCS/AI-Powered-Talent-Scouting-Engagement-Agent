import os
from typing import List, Dict, Any, Optional
import json
import requests

from src.agent_architecture import CandidateRecord, CandidateState, JDMetadata


class PeopleDataLabsScout:
    """Discover candidates using People Data Labs with flexible Elasticsearch filters."""

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

        # 1. Broad Title Logic: Agar title unknown hai toh default "Software Engineer" use karega
        title = jd.title.strip() if jd.title and jd.title.lower(
        ) != "unknown" else "Software Engineer"

        # 2. Elasticsearch Query Structure
        # 'must' matlab ye hona hi chahiye (Job Title)
        # 'should' matlab ye ho toh achha hai (Skills, Location, Experience) - ye results block nahi karta
        query = {
            "bool": {
                "must": [
                    {"match": {"job_title": title}}
                ],
                "should": []
            }
        }

        # 3. Skills Flexibility: Inhe 'should' mein daala hai taaki 100% match zaroori na ho
        if jd.skills:
            unique_skills = [s.lower().strip()
                             for s in jd.skills[:10] if isinstance(s, str)]
            if unique_skills:
                query["bool"]["should"].append(
                    {"terms": {"skills": unique_skills}})

        # 4. Location Flexibility
        if jd.location and jd.location.lower() not in {"global", "unknown"}:
            normalized = jd.location.lower()
            if "remote" in normalized:
                query["bool"]["should"].append(
                    {"match": {"location_name": "Remote"}})
            else:
                query["bool"]["should"].append(
                    {"match": {"location_name": jd.location}})

        # 5. Experience Range
        if jd.experience_years > 0:
            query["bool"]["should"].append({
                "range": {"inferred_years_experience": {"gte": jd.experience_years}}
            })

        # PDL expects the query as a JSON string
        payload: Dict[str, Any] = {
            "query": json.dumps(query),
            "dataset": "resume",
            "size": effective_limit
        }

        # Additional filters agar metadata mein kuch aur ho
        if jd.additional_filters:
            payload.update(jd.additional_filters)

        return payload

    def _execute_search(self, payload: Dict[str, Any]) -> requests.Response:
        return self.session.post(self.BASE_URL, json=payload, timeout=30)

    def _build_relaxed_search_payload(self, jd: JDMetadata, limit: int) -> Dict[str, Any]:
        # Fallback: Agar results bilkul nahi milte toh skills criteria hata do
        payload = self.build_search_payload(jd, limit=limit)
        # Re-parsing query string to remove skills and re-stringifying
        q_obj = json.loads(payload["query"])
        # Remove skills term if it exists in should
        q_obj["bool"]["should"] = [s for s in q_obj["bool"]["should"]
                                   if "terms" not in s or "skills" not in s["terms"]]
        payload["query"] = json.dumps(q_obj)
        return payload

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
        raw_profiles = data.get("data", [])

        if debug:
            print(
                f"\n[PDL returned {len(raw_profiles)} profiles on initial search]")

        # Fallback logic agar 1 se kam result mile
        if len(raw_profiles) <= 1:
            if debug:
                print("Low results found, trying relaxed search...")
            relaxed_payload = self._build_relaxed_search_payload(
                jd, limit=limit)
            fallback_response = self._execute_search(relaxed_payload)
            if fallback_response.status_code == 200:
                fallback_data = fallback_response.json()
                fallback_profiles = fallback_data.get("data", [])
                if len(fallback_profiles) > len(raw_profiles):
                    raw_profiles = fallback_profiles
                    if debug:
                        print(
                            f"[Relaxed search returned {len(raw_profiles)} profiles]")

        return raw_profiles

    def normalize_candidate(
        self,
        raw_profile: Dict[str, Any],
        search_payload: Optional[Dict[str, Any]] = None,
    ) -> CandidateRecord:
        metadata = {
            "raw_profile": json.dumps(raw_profile)
        }
        if search_payload is not None:
            metadata["search_payload"] = json.dumps(search_payload)

        experience_years = raw_profile.get("inferred_years_experience")
        if experience_years is None:
            experience_years = raw_profile.get("experience_years", 0)

        raw_skills = raw_profile.get(
            "job_skills") or raw_profile.get("skills") or []
        if isinstance(raw_skills, str):
            top_skills = [raw_skills]
        elif isinstance(raw_skills, list):
            top_skills = [str(skill).strip()
                          for skill in raw_skills if skill is not None]
        else:
            top_skills = []

        return CandidateRecord(
            candidate_id=str(
                raw_profile.get("id", raw_profile.get("person_id", "unknown"))
            ),
            name=raw_profile.get("full_name", "Unknown"),
            role=raw_profile.get("job_title", "Unknown"),
            experience_years=int(experience_years or 0),
            top_skills=top_skills,
            location=raw_profile.get("location_name", "Unknown"),
            source="PeopleDataLabs",
            current_state=CandidateState.DISCOVERED,
            metadata=metadata,
        )

    def discover_candidates(self, jd: JDMetadata, limit: int = 50, debug: bool = False) -> List[CandidateRecord]:
        raw_profiles = self.search_candidates(jd, limit=limit, debug=debug)

        payload = self.build_search_payload(jd, limit=limit)
        candidates = [
            self.normalize_candidate(profile, search_payload=payload)
            for profile in raw_profiles
        ]
        return candidates
