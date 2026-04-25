# AI-Powered Talent Scouting & Engagement Prototype

## Project Overview

This repository contains a working local prototype for a talent scouting pipeline.
It parses a job description, discovers candidates via People Data Labs, creates embeddings,
stores candidate vectors in Pinecone, and returns ranked candidate matches.

The main user interface is a Streamlit app at `streamlit_app.py`.

## What the prototype does

1. Parses a job description from `.txt`, `.pdf`, or pasted text using Affinda.
2. Builds a candidate search payload from the extracted JD metadata.
3. Calls People Data Labs to discover candidate profiles.
4. Converts candidate text into embeddings with `sentence-transformers/all-MiniLM-L6-v2`.
5. Stores candidate embeddings in Pinecone and queries for the best semantic matches.
6. Simulates candidate outreach and computes a final score using match and interest.

## Repo structure

- `streamlit_app.py` — Streamlit frontend for uploading JDs, entering text, and viewing results.
- `.env.example` — sample environment variables.
- `.env` — local secrets file (not checked into git).
- `requirements.txt` — project dependencies.
- `src/` — core application modules:
  - `src/orchestrator.py` — main workflow orchestration.
  - `src/jd_analyst.py` — Affinda JD parsing + fallback heuristic parser.
  - `src/scout_agent.py` — People Data Labs candidate discovery.
  - `src/embeddings.py` — text embeddings using Hugging Face transformers.
  - `src/matching_engine.py` — Pinecone embeddings storage and query.
  - `src/agent_architecture.py` — data models, score calculator, and state types.

## Key components

### JD parsing
- Uses Affinda API with `.txt` or `.pdf` upload.
- Extracts job title, skills, location, experience, seniority, and description.
- If Affinda fails, it uses a fallback heuristic parser.

### Candidate discovery
- Uses People Data Labs search API.
- Builds filters from JD metadata and returns candidate records.

### Embeddings
- Uses Hugging Face `sentence-transformers/all-MiniLM-L6-v2`.
- Embedding dimension is `384`.
- Normalizes vectors before sending to Pinecone.

### Matching
- Stores candidate embeddings in a Pinecone serverless index.
- Queries the index with the job description vector.
- Filters for candidates in the `Discovered` state.

### Engagement scoring
- `match_score` comes from Pinecone similarity.
- `interest_score` is assigned by the engagement bot.
- Final score is a weighted sum:

```python
final_score = 0.7 * match_score + 0.3 * interest_score
```

## Environment setup

Copy `.env.example` to `.env` and fill in the keys:

```dotenv
PDL_API_KEY=your_pdl_api_key_here
AFFINDA_API_KEY=your_affinda_api_key_here
HF_TOKEN=your_huggingface_token_here
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=us-east-1
PINECONE_INDEX=talent-matching-index
PINECONE_POD_TYPE=serverless
```

> If you are only using the local Hugging Face model download, `HF_TOKEN` is optional but recommended for faster access.

## Local setup and run

1. Open terminal in `d:\DeccanAI`.
2. Create and activate the Python virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies:

```powershell
pip install -r requirements.txt
```

4. Start the Streamlit app:

```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

5. Open the browser at `http://localhost:8501`.

## How to use the app

- Upload a job description file (`.txt` or `.pdf`) or paste JD text directly.
- Set candidate limit and top-K matches.
- Click `Process JD`.
- The UI will show the parsed result and ranked candidate JSON output.

## Sample input

```text
Job Title: Senior AI Engineer
Location: Remote
Experience: 5+ years
Skills: Python, NLP, transformers, embeddings, Pinecone
Description: Build a talent scouting agent that parses job descriptions, searches candidate profiles, and ranks best fits.
```

## Sample output

```json
[
  {
    "candidate_id": "eBLH6swnCE85Xr61cUi9rQ_0000",
    "name": "ben margolis",
    "role": "senior cloud infrastructure engineer",
    "match_score": 0.4835,
    "interest_score": 1.0,
    "final_score": 0.6385,
    "state": "Engaged",
    "response_text": "Hi ben margolis, thank you for reviewing this opportunity for SDE 1..."
  },
  {
    "candidate_id": "ZFI6tUYfR8qSnG0cp5KN7Q_0000",
    "name": "stephen lanford",
    "role": "it manager, application system engineering (ase)",
    "match_score": 0.3343,
    "interest_score": 1.0,
    "final_score": 0.5198,
    "state": "Engaged"
  }
]
```

## Architecture summary

- `streamlit_app.py` handles the UI and user interactions.
- `src/orchestrator.py` glues the workflow together.
- `src/jd_analyst.py` parses the JD.
- `src/scout_agent.py` searches candidate profiles.
- `src/embeddings.py` generates text embeddings.
- `src/matching_engine.py` upserts/query vectors in Pinecone.
- `src/agent_architecture.py` defines the candidate and JD data models.

## Notes for deliverables

- Use this README for your public repo documentation.
- Include a short 3–5 minute demo showing the Streamlit app, JD upload, and output.
- Add a simple architecture diagram or bullet list describing the flow.
- Keep the deployment step separate; this repo is ready as a local prototype.


## Environment Variables

Your `.env` file should include:

```dotenv
PDL_API_KEY=your_peopledatalabs_key
AFFINDA_API_KEY=your_affinda_key
PINECONE_API_KEY=pcsk_4NcPRY_4hUMk4LmKPwLHp9Yp8ap54YsSqXbjHgptRUvxTd9HCJA3jDgebM78RqzHCCNwmJ
PINECONE_ENVIRONMENT=us-east-1
PINECONE_INDEX=talent-matching-index
PINECONE_POD_TYPE=serverless
```

## Notes

- The orchestration flow uses Pinecone serverless for vector storage.
- Candidate states move from `Discovered` to `Matched` to `Engaged`.
- The final score combines semantic match strength and engagement interest.
