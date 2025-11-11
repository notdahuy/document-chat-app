import streamlit as st
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai.errors import APIError
import os
import io
from dotenv import load_dotenv

# Import functions from vector_store.py
from vector_store import *

# Load environment variables
load_dotenv()

# --- I18N DATA DICTIONARY ---
LANG_DICT = {
    "Vietnamese": {
        "title": "🤖 Chatbot Hỏi Đáp Tài Liệu PDF (RAG System)",
        "sidebar_header": "⚙️ Settings",
        "lang_select": "Chọn Ngôn Ngữ Giao Diện (UI Language):",
        "upload_label": "📤 Step 1: Upload File PDF or DOCX (Multiple files supported)",
        "query_label": "💬 Step 2: Enter your question about the document(s)",
        "query_placeholder": "Ví dụ: Tóm tắt nội dung chính của tài liệu này?",
        "response_lang_label": "🗣️ Step 3: Select the response language (LLM)",
        "button_text": "🔍 Find Answer",
        "status_processing": "Retrieving context and generating answer...",
        "status_success": "Successfully processed",
        "answer_header": "✅ Answer:",
        "context_expander": "📖 Source References (Retrieved Context)",
        "error_no_file": "Please upload PDF/DOCX file(s) before asking.",
        "error_no_query": "Vui lòng nhập câu hỏi.",
        "error_llm_key": "LLM connection error. Please check GEMINI_API_KEY in .env.",
        "warning_no_index": "Cannot create index. Please check the PDF file(s).",
    },
    "English": {
        "title": "🤖 PDF Document Q&A Chatbot (RAG System)",
        "sidebar_header": "⚙️ Settings",
        "lang_select": "Select UI Language:",
        "upload_label": "📤 Step 1: Upload PDF or DOCX file(s) (Multiple files supported)",
        "query_label": "💬 Step 2: Enter your question about the document(s)",
        "query_placeholder": "Example: Summarize the main contents of this document?",
        "response_lang_label": "🗣️ Step 3: Select the response language (LLM)",
        "button_text": "🔍 Find Answer",
        "status_processing": "Retrieving context and generating answer...",
        "status_success": "Successfully processed",
        "answer_header": "✅ Answer:",
        "context_expander": "📖 Source References (Retrieved Context)",
        "error_no_file": "Please upload PDF/DOCX file(s) before asking.",
        "error_no_query": "Please enter a question.",
        "error_llm_key": "LLM connection error. Please check GEMINI_API_KEY in .env.",
        "warning_no_index": "Cannot create index. Please check the PDF file(s).",
    }
}


# Load base resources (Model and Client) only once
@st.cache_resource
def load_base_resources():
    """Loads SentenceTransformer Model and Gemini Client."""
    model = SentenceTransformer('all-MiniLM-L6-v2')

    client = None
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            st.error("GEMINI_API_KEY not found in environment variables.")
        else:
            client = genai.Client(api_key=api_key)
    except Exception as e:
        st.error(f"Error while connecting to Gemini: {e}")

    return model, client


# Create Vector Store for multiple uploaded files
@st.cache_resource
def create_vector_store_for_upload(uploaded_files, _model):
    """Processes multiple uploaded files, creates combined chunks, and a single FAISS index."""
    all_chunks = []

    # Process each uploaded file
    for uploaded_file in uploaded_files:
        try:
            file_bytes = io.BytesIO(uploaded_file.read())

            if uploaded_file.type == "application/pdf":
                text = extract_text_from_pdf(file_bytes)
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                text = extract_text_from_docx(file_bytes)
            else:
                st.warning(f"Skipped unsupported file type: {uploaded_file.name}")
                continue

            if not text.strip():
                st.warning(f"File {uploaded_file.name} is empty.")
                continue

            # Combine chunks from all files
            all_chunks.extend(chunk_text(text))

        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")

    if not all_chunks:
        return None, None

    # Create Embeddings for all chunks
    embeddings = _model.encode(all_chunks)
    embeddings = np.array(embeddings).astype('float32')

    # Create FAISS Index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index, all_chunks


# Main RAG function
def retrieve_and_generate(query, model, index, chunks, client, k=15, language="Vietnamese"):
    """Performs retrieval and generation using Gemini LLM."""
    if client is None:
        return "Error: Cannot connect to Gemini LLM. Please check your API key.", ""

    # 1. Query Encoding
    query_vector = model.encode([query]).astype('float32')
    # 2. Retrieval
    distances, indices = index.search(query_vector, k)
    retrieved_chunks = [chunks[i] for i in indices[0]]
    context = "\n\n---\n\n".join(retrieved_chunks)

    # 3. Generation Prompt (Injecting response language)
    system_prompt = (
        f"You are a professional document Q&A assistant. Your task is to answer "
        f"the user's question **ONLY based on the CONTEXT** provided. "
        f"The answer MUST be written in **{language}**. "
        "If the answer is not in the context, respond: 'Tôi xin lỗi, thông tin này "
        "không có trong tài liệu nguồn.' (or the equivalent in the requested language)."
    )
    user_content_base = f"CONTEXT:\n{context}\n\n---\n\nQUESTION: {query}"
    full_prompt = f"{system_prompt}\n\n[RAG Data]\n\n{user_content_base}"

    # Call API
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                {"role": "user", "parts": [{"text": full_prompt}]}
            ]
        )
        return response.text, context

    except APIError as e:
        return f"Gemini API Error: {e}", context
    except Exception as e:
        return f"An unexpected error occurred: {e}", context


# --- Streamlit App Configuration ---
st.set_page_config(page_title="RAG Chatbot", layout="wide")

# Sidebar for UI Language Selection
with st.sidebar:
    st.header(LANG_DICT["Vietnamese"]["sidebar_header"])

    ui_language = st.selectbox(
        LANG_DICT["Vietnamese"]["lang_select"],
        options=list(LANG_DICT.keys()),
        index=0
    )

# Get the localized text dictionary
lang = LANG_DICT[ui_language]

# Load base resources
model, client = load_base_resources()
index, chunks = None, None

# --- MAIN APP LAYOUT ---
st.title(lang["title"])
st.markdown("---")

# 1. File Upload Area (Accepts multiple files)
uploaded_files = st.file_uploader(
    lang["upload_label"],
    type=["pdf", "docx"],
    accept_multiple_files=True
)

if uploaded_files:
    # Pass all uploaded files to the processor
    with st.spinner(f"Processing {len(uploaded_files)} file(s)..."):
        index, chunks = create_vector_store_for_upload(uploaded_files, model)

    if index and chunks:
        st.success(f"{lang['status_success']} {len(chunks)} chunks.")
    else:
        st.warning(lang["warning_no_index"])

# --- Q&A AREA ---
st.markdown("---")

# 2. Query Input
query = st.text_input(
    lang["query_label"],
    placeholder=lang["query_placeholder"]
)

# 3. Response Language Selection
selected_language = st.selectbox(
    lang["response_lang_label"],
    options=["Vietnamese", "English", "日本語 (Japanese)", "Español (Spanish)"],
    index=0
)

if st.button(lang["button_text"]):
    if not uploaded_files:
        st.warning(lang["error_no_file"])
    elif not query:
        st.warning(lang["error_no_query"])
    elif client is None:
        st.error(lang["error_llm_key"])
    else:
        with st.spinner(lang["status_processing"]):
            # Pass the selected language to the RAG function
            answer, context = retrieve_and_generate(
                query,
                model,
                index,
                chunks,
                client,
                language=selected_language
            )

            # Display results
            st.subheader(lang["answer_header"])
            st.info(answer)

            with st.expander(lang["context_expander"]):
                st.text(context)