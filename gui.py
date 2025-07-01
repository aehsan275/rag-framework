import streamlit as st
from rag import rag
import html

tabs = st.tabs(["Model", "Logs"])

with tabs[0]:
    st.title("Retrieval Augmented Generation Model")
    user_input = st.text_input("Ask a question:", "")
    info = st.empty()

    if st.button("Submit"):
        info.empty()
        info.info(rag.rag_response(user_input.title()))

with tabs[1]:
    st.title("Logs")

    if st.button("Clear logs"):
        with open("logs.log", "w") as file:
            pass
        st.rerun()
    info = st.empty()
    logs = open("logs.log", "r")
    log_history = html.escape(logs.read()).replace("\n", "<br>")
    html_text = f"""
    <p style="font-family: 'Courier New', monospace; font-size: 14px; color: white;">
        {log_history}
    </p>
    """
    st.markdown(html_text, unsafe_allow_html=True)
    logs.close()


