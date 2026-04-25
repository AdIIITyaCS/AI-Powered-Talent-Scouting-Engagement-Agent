import streamlit as st
from transformers import logging
from src.orchestrator import TalentScoutingOrchestrator

logging.set_verbosity_error()

st.set_page_config(page_title="Talent Scouting UI", layout="centered")
st.title("Talent Scouting Interface")
st.markdown(
    "Upload your job description as `.txt` or `.pdf`, or paste the text directly. "
    "The backend will parse the JD, fetch candidates, create embeddings, and return ranked matches."
)

uploaded_file = st.file_uploader("Upload JD file", type=["txt", "pdf"])
text_input = st.text_area("Or paste JD text here", height=250)

candidate_limit = st.number_input(
    "Candidate limit", min_value=1, max_value=50, value=8)
top_k = st.number_input("Top k matches", min_value=1, max_value=20, value=5)
debug = st.checkbox("Show debug in backend terminal", value=False)
process = st.button("Process JD")

status = st.empty()
result_area = st.empty()

if process:
    if not uploaded_file and not text_input.strip():
        st.error("Please upload a JD file or paste the JD text.")
    else:
        orchestrator = TalentScoutingOrchestrator()
        try:
            # 1) Parse JD with Affinda
            if uploaded_file:
                status.text("Parsing JD file with Affinda...")
                print("JD uploaded, parsing via Affinda...")
                file_bytes = uploaded_file.read()
                jd_metadata = orchestrator.jd_analyzer.parse_job_description_from_bytes(
                    file_bytes, uploaded_file.name, debug=debug
                )
            else:
                status.text("Parsing JD text with Affinda...")
                print("Parsing pasted JD via Affinda...")
                jd_metadata = orchestrator.jd_analyzer.parse_job_description(
                    text_input, debug=debug
                )

            status.text("JD parsed (Affinda)")
            print("JD parsed (Affinda)")

            # 2) Run the rest of the orchestration
            status.text("Fetching candidates from PDL...")
            print("Candidates fetched (PDL)")
            results = orchestrator._run_with_metadata(
                jd_metadata, candidate_limit, top_k, debug
            )

            status.text("Embeddings created")
            print("Embeddings created")
            status.text("Matching completed")
            print("Matching completed")

            status.success("Processing complete!")
            if results:
                result_area.json(results)
            else:
                result_area.info("No matches found.")
        except Exception as exc:
            status.error("Processing failed.")
            st.exception(exc)
