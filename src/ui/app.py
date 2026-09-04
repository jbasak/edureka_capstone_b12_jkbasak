"""Streamlit chat and document upload interface."""

import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import streamlit as st

st.set_page_config(page_title="Procurement intelligence", page_icon=":material/search:", layout="wide")
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")


def api_request(path: str, method: str = "GET", payload: bytes | None = None, content_type: str | None = None) -> dict:
    headers = {"Content-Type": content_type} if content_type else {}
    request = Request(f"{API_BASE_URL}{path}", data=payload, headers=headers, method=method)
    try:
        with urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Backend returned HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(
            f"Cannot connect to the FastAPI backend at {API_BASE_URL}. "
            "Start it with: uvicorn src.api.main:app --reload"
        ) from exc


def upload_file(uploaded_file: object) -> dict:
    boundary = "----ProcurementAssistantBoundary"
    body = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{uploaded_file.name}\"\r\n"
        f"Content-Type: application/octet-stream\r\n\r\n"
    ).encode() + uploaded_file.getvalue() + f"\r\n--{boundary}--\r\n".encode()
    return api_request("/documents", "POST", body, f"multipart/form-data; boundary={boundary}")


if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Knowledge base")
    st.caption("Upload procurement evidence before asking a question.")
    uploads = st.file_uploader("Documents", type=["pdf", "txt", "csv", "xlsx"], accept_multiple_files=True)
    if st.button("Index documents", type="primary", icon=":material/upload:", width="stretch"):
        if not uploads:
            st.warning("Select at least one document.")
        else:
            for uploaded in uploads:
                try:
                    result = upload_file(uploaded)
                    st.success(f"{result['filename']}: {result['chunks_stored']} chunks indexed")
                except (RuntimeError, ValueError, KeyError) as exc:
                    st.error(f"Upload failed: {exc}")
    st.caption(f"Backend: {API_BASE_URL}")

st.title("Procurement intelligence")
st.write("Ask grounded questions across RFP responses, pricing sheets, and compliance evidence.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Compare annual pricing, compliance, or vendor terms"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant", avatar=":material/verified:"):
        try:
            result = api_request("/query", "POST", json.dumps({"prompt": prompt}).encode(), "application/json")
            answer = result["answer"]
            st.markdown(answer)
        except (RuntimeError, ValueError, KeyError) as exc:
            answer = f"Request failed: {exc}"
            st.error(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})