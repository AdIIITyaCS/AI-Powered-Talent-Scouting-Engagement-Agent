import io
import json
import os
import re
from typing import Dict, Any, Optional

import requests
from dotenv import load_dotenv

from src.agent_architecture import JDMetadata

load_dotenv()


class AffindaJDAnalyst:
    """Parse job descriptions using Affinda or a fallback heuristic."""

    ENDPOINT = "https://api.affinda.com/v2/job_descriptions"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("AFFINDA_API_KEY")
        if not self.api_key:
            raise ValueError("AFFINDA_API_KEY is required for JD parsing")
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
        }

    def parse_job_description(self, jd_text: str, debug: bool = False) -> JDMetadata:
        return self.parse_job_description_from_bytes(
            jd_text.encode("utf-8"), "job_description.txt", debug=debug
        )

    def parse_job_description_from_bytes(
        self, file_bytes: bytes, filename: str, debug: bool = False
    ) -> JDMetadata:
        mime_type = "text/plain" if filename.lower().endswith(".txt") else "application/pdf"
        file_obj = io.BytesIO(file_bytes)
        files = {
            "file": (filename, file_obj, mime_type),
        }
        form_data = {
            "wait": "true",
        }

        try:
            response = requests.post(
                self.ENDPOINT,
                files=files,
                data=form_data,
                headers=self.headers,
                timeout=60,
            )
            if debug:
                print("\n[Affinda endpoint]", self.ENDPOINT)
                print("[Affinda request status]", response.status_code)
                print("[Affinda response text]", response.text[:2000])

            response.raise_for_status()
            result = response.json()
            if debug:
                print("\n[Affinda JD JSON]\n", json.dumps(
                    result, indent=2, ensure_ascii=False)[:3000])
            return self._map_affinda_response(result, file_bytes.decode("utf-8", errors="ignore"))

        except Exception as exc:
            if debug:
                print("[Affinda parse failed]", repr(exc))
            return self._heuristic_parse(file_bytes.decode("utf-8", errors="ignore"))

    def _map_affinda_response(self, response: Dict[str, Any], jd_text: str) -> JDMetadata:
        data = response.get("data", {})

        # --- Skills ---
        raw_skills = data.get("skills", []) or []
        skills = []
        for skill in raw_skills:
            if isinstance(skill, dict):
                name = skill.get("name", "") or skill.get("parsed", "")
                if name:
                    skills.append(name)
            elif isinstance(skill, str):
                skills.append(skill)

        # --- Job Title ---
        job_title_obj = data.get("jobTitle", {}) or {}
        title = ""
        if isinstance(job_title_obj, dict):
            title = job_title_obj.get(
                "parsed", "") or job_title_obj.get("raw", "")
        elif isinstance(job_title_obj, str):
            title = job_title_obj
        if not title:
            title = self._extract_field(jd_text, ["title", "role"])

        # --- Location ---
        location_obj = data.get("location", {}) or {}
        location = ""
        if isinstance(location_obj, dict):
            location = location_obj.get(
                "formatted", "") or location_obj.get("rawInput", "")
        elif isinstance(location_obj, str):
            location = location_obj
        if not location:
            location = self._extract_field(
                jd_text, ["location", "city", "remote"])
        location = self._normalize_location(location)

        # --- Experience ---
        experience_years = 0
        years_exp = data.get("yearsExperience", None)
        if years_exp is not None:
            try:
                if isinstance(years_exp, dict):
                    experience_years = int(years_exp.get("parsed", 0) or 0)
                else:
                    experience_years = int(years_exp)
            except (ValueError, TypeError):
                experience_years = 0

        # --- Description (use rawText from Affinda if available, else original) ---
        raw_text = data.get("rawText", "") or ""
        description = raw_text if raw_text.strip() else jd_text

        # --- Industry (from organization name as a proxy) ---
        org_obj = data.get("organizationName", {}) or {}
        industry = ""
        if isinstance(org_obj, dict):
            industry = org_obj.get("parsed", "") or ""
        elif isinstance(org_obj, str):
            industry = org_obj

        # --- Seniority ---
        seniority = ""
        seniority_obj = data.get("seniority", None)
        if isinstance(seniority_obj, dict):
            seniority = seniority_obj.get("parsed", "") or ""
        elif isinstance(seniority_obj, str):
            seniority = seniority_obj

        return JDMetadata(
            title=title,
            skills=skills,
            seniority=seniority,
            location=location,
            description=description,
            industry=industry,
            experience_years=experience_years,
            additional_filters={},
        )

    def _heuristic_parse(self, jd_text: str) -> JDMetadata:
        title = self._extract_field(
            jd_text, ["title", "role", "position"]) or "Unknown"
        location = self._extract_field(
            jd_text, ["location", "remote", "city"]) or "Global"
        location = self._normalize_location(location)
        skills = self._extract_skills(jd_text)
        experience_years = self._extract_experience(jd_text)

        return JDMetadata(
            title=title,
            skills=skills,
            seniority="",
            location=location,
            description=jd_text,
            industry="",
            experience_years=experience_years,
            additional_filters={},
        )

    def _normalize_location(self, location: str) -> str:
        if not location:
            return ""
        normalized = location.strip().lower()
        if "remote" in normalized:
            return "Remote"
        if "hybrid" in normalized:
            return "Hybrid"
        if "us-based" in normalized or "u.s." in normalized or "usa" in normalized or "united states" in normalized:
            return "United States"
        return location.strip()

    def _extract_field(self, text: str, keys: list[str]) -> str:
        for key in keys:
            match = re.search(
                rf"{key}[:\-]\s*([A-Za-z0-9\s,-]+)", text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return ""

    def _extract_skills(self, text: str) -> list[str]:
        skill_section = re.search(
            r"skills[:\-][\s\S]+?(?=\n\n|responsibilities|requirements|$)", text, re.IGNORECASE)
        if not skill_section:
            return []
        candidates = re.findall(r"[A-Za-z+#\.]+", skill_section.group(0))
        return [skill.strip() for skill in candidates if len(skill.strip()) > 1][:20]

    def _extract_experience(self, text: str) -> int:
        match = re.search(r"(\d+)\+?\s+years?", text, re.IGNORECASE)
        return int(match.group(1)) if match else 0
