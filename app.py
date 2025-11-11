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
from lang import lang_dict

# Load environment variables
load_dotenv()

# Load base resources (Model and Client) only once
@st.cache_resource
def load_base_resources():
    """Loads SentenceTransformer Model and Gemini Client."""
    model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

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


# Create Vector Store for the single uploaded file
@st.cache_resource
def create_vector_store_for_upload(uploaded_file, _model):
    """Processes a single uploaded file, creates chunks, and a single FAISS index."""

    try:
        file_bytes = io.BytesIO(uploaded_file.read())

        if uploaded_file.type == "application/pdf":
            text = extract_text_from_pdf(file_bytes)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            text = extract_text_from_docx(file_bytes)
        else:
            st.error("Unsupported file type.")
            return None, None

        if not text.strip():
            st.warning(f"File {uploaded_file.name} is empty.")
            return None, None

        # Chunk the text
        all_chunks = chunk_text(text)

        # Create Embeddings for all chunks
        embeddings = _model.encode(all_chunks)
        embeddings = np.array(embeddings).astype('float32')

        # Create FAISS Index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        return index, all_chunks

    except Exception as e:
        st.error(f"Error processing {uploaded_file.name}: {e}")
        return None, None


# Main RAG function
def retrieve_and_generate(query, model, index, chunks, client, k=15):
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
        "You are a professional document Q&A assistant. Your task is to answer "
        "the user's question **ONLY based on the CONTEXT** provided. "
        "The answer MUST be written in the **SAME LANGUAGE as the user's QUESTION**. "
        "If the answer is not in the context, you must respond with a polite disclaimer "
        "stating that the information is not found in the source documents, using the "
        "**SAME LANGUAGE as the user's question**. Do not invent information."
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
    st.header(lang_dict["Vietnamese"]["sidebar_header"])

    ui_language = st.selectbox(
        lang_dict["Vietnamese"]["lang_select"],
        options=list(lang_dict.keys()),
        index=0
    )

# Get the localized text dictionary
lang = lang_dict[ui_language]

# Load base resources
model, client = load_base_resources()
index, chunks = None, None

# Main app layout
st.title(lang["title"])
st.markdown("---")

# 1. File Upload Area (Single file only)
uploaded_file = st.file_uploader(
    lang["upload_label"],
    type=["pdf", "docx"],
    # Removed accept_multiple_files=True
)

if uploaded_file:
    # Pass the single uploaded file to the processor
    with st.spinner(f"Processing {uploaded_file.name}..."):
        index, chunks = create_vector_store_for_upload(uploaded_file, model)

# --- Q&A AREA ---
st.markdown("---")

# 2. Query Input
query = st.text_input(
    lang["query_label"],
    placeholder=lang["query_placeholder"]
)


if st.button(lang["button_text"]):
    # Changed uploaded_files check to uploaded_file
    if not uploaded_file:
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
            )

            # Display results
            st.subheader(lang["answer_header"])
            st.info(answer)

            with st.expander(lang["context_expander"]):
                st.text(context)