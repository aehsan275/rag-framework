import streamlit as st
from rag import rag
import html

tabs = st.tabs(["Model", "Logs"]) # two tabs, one for model and one for logs

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
        st.rerun() # refreshes page
    info = st.empty()
    reader = open("logs.log", "r") # rereads logs each time the page is opened
    log_history = html.escape(reader.read()).replace("\n", "<br>") # allows logs to be safely used in html
    html_text = f"""
    <p style="font-family: 'Courier New', monospace; font-size: 14px; color: white;">
        {log_history}
    </p>
    """ # html for logs
    st.markdown(html_text, unsafe_allow_html=True)
    reader.close()


