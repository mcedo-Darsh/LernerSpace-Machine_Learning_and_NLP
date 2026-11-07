import os
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rag import RagPipeline 

st.set_page_config(page_title="IITB Insti-Assist", page_icon="🎓", layout="centered")


@st.cache_resource(show_spinner="Loading index and embedding model...")
def load_pipeline():
    return RagPipeline()


st.title("🎓 IITB Insti-Assist")
st.caption("Academic Assistant — ask about course registration, grading policy, the academic calendar, exam rules, and course offerings at IIT Bombay.")

if not os.environ.get("GEMINI_API_KEY"):
    st.error("GEMINI_API_KEY is not set.")
    st.stop()

try:
    pipeline = load_pipeline()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("About")
    st.write("This assistant answers only from IIT Bombay's official academic documents: the Academic Calendar 2026-27, the Autumn timetable, the grading policy, and the department-wise course lists. If the answer isn't in those documents, it will say so instead of guessing.")
    top_k = st.slider("Chunks to retrieve (k)", min_value=2, max_value=10, value=5)
    if st.button("Clear conversation"):
        st.session_state.history = []
        st.rerun()

# replay history
for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn.get("sources"):
            with st.expander("Sources"):
                for s in turn["sources"]:
                    st.markdown(f"- **{s['source']}**, page {s['page']} (score {s['score']:.2f})")

query = st.chat_input("Ask about IIT Bombay academics...")

if query:
    st.session_state.history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving relevant documents and generating an answer..."):
            result = pipeline.answer(query, k=top_k)
        st.markdown(result["answer"])
        if result["sources"]:
            with st.expander("Sources"):
                for s in result["sources"]:
                    st.markdown(f"- **{s['source']}**, page {s['page']} (score {s['score']:.2f})")

    st.session_state.history.append(
        {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
    )
