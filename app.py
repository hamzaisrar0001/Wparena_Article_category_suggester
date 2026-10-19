import streamlit as st
from classifier import train, suggest

st.set_page_config(page_title="Article Category Suggester", page_icon="🏷️")
st.title("🏷️ Article Category Suggester")
st.caption("Paste a WordPress / tech article title and get a suggested category.")


@st.cache_resource
def get_model():
    return train()


title = st.text_input("Article title", placeholder="e.g. How to harden your WordPress login page")
if title.strip():
    results = suggest(get_model(), title)
    st.subheader(f"Suggested category: {results[0][0]}")
    for cat, p in results:
        st.progress(float(p), text=f"{cat} — {p:.0%}")
