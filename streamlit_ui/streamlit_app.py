import streamlit as st
import requests
import os
import socket
import requests

API_URL = os.getenv("DOCHELPER_API_URL", "http://0.0.0.0:8000/v2/ask_ai")
UPLOAD_URL = os.getenv("DOCHELPER_UPLOAD_URL", "http://0.0.0.0:8000/core/upload-pdf")
print("hostbyname", socket.gethostbyname("dochelper.railway.internal"))
print("hostbyname:8000", socket.gethostbyname("dochelper.railway.internal:8000"))


ipv4 = socket.gethostbyname("dochelper.railway.internal")
url = f"http://{ipv4}:8000/v2/ask_ai"
API_URL = f"http://{ipv4}:8000/v2/ask_ai"
UPLOAD_URL = f"http://{ipv4}:8000/core/upload-pdf"

st.set_page_config(page_title="docHelper Q&A", layout="centered")
st.title("📄 docHelper Q&A")

# --- State for upload/Q&A mode ---
if "show_upload" not in st.session_state:
    st.session_state.show_upload = False
if "answer" not in st.session_state:
    st.session_state.answer = None
if "context" not in st.session_state:
    st.session_state.context = None
if "sources" not in st.session_state:
    st.session_state.sources = None
if "question_input" not in st.session_state:
    st.session_state.question_input = None
if "question_input_key" not in st.session_state:
    st.session_state.question_input_key = 0


def reset_qa():
    st.session_state.question_input = None
    st.session_state.answer = None
    st.session_state.context = None
    st.session_state.sources = None
    st.session_state.question_input_key += 1


def go_to_upload():
    st.session_state.show_upload = True
    reset_qa()


def return_to_qna():
    st.session_state.show_upload = False
    reset_qa()


# --- Top right Upload button ---
col1, col2 = st.columns([4, 1])
with col2:
    st.button(
        "Upload PDF",
        on_click=go_to_upload,
        key="upload_btn",
        help="Upload a new PDF",
        icon="🔄",
    )

# --- Upload Screen ---
if st.session_state.show_upload:
    st.markdown("### Upload a PDF")
    uploaded_file = st.file_uploader(
        "Choose a PDF file", type=["pdf"], key="pdf_uploader"
    )
    if uploaded_file is not None:
        with st.spinner("Uploading PDF..."):
            files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
            response = requests.post(UPLOAD_URL, files=files)
            if response.status_code == 200:
                st.success(f"Uploaded: {uploaded_file.name}")
            else:
                st.error(f"Upload failed: {response.text}")
    st.button("Return to Q&A", on_click=return_to_qna, key="return_btn")
    st.stop()

# --- Q&A Form (always visible) ---
with st.form("ask_form"):
    question = st.text_area(
        "Ask a question about your PDFs:",
        height=100,
        value=st.session_state.question_input
        if st.session_state.question_input is not None
        else "",
        key=f"question_area_{st.session_state.question_input_key}",
    )
    submitted = st.form_submit_button("Get Answer")
    if submitted and question:
        with st.spinner("Thinking..."):
            payload = {"question": question}
            try:
                resp = requests.post(API_URL, params=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    st.session_state.answer = data.get("answer", "No answer returned.")
                    st.session_state.context = data.get("context_chunks", None)
                    st.session_state.sources = data.get("sources", None)
                    st.session_state.question_input = question
                else:
                    st.error(f"Error: {resp.text}")
            except Exception as e:
                st.error(f"Request failed: {e}")

# --- Answer Display and Ask Another Question Button ---
if st.session_state.answer is not None:
    st.markdown("#### Answer:")
    st.write(st.session_state.answer)
    if st.session_state.sources:
        st.markdown("**Sources:**")
        st.write(", ".join(st.session_state.sources))
    st.markdown("<a id='answer'></a>", unsafe_allow_html=True)
    st.markdown(
        """
        <script>
        window.location.hash = 'answer';
        </script>
    """,
        unsafe_allow_html=True,
    )
    st.button("Ask another question", key="ask_another_btn", on_click=reset_qa)
