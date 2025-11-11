import numpy as np
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from docx import Document


# Read and chunk text
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    all_text = ""

    for page in reader.pages:
        text = page.extract_text()
        if text:
            all_text += text + "\n"

    return all_text

def extract_text_from_docx(docx_path):
    document = Document(docx_path)
    full_text = []
    for paragraph in document.paragraphs:
        full_text.append(paragraph.text)
    return "\n".join(full_text)

def chunk_text(text, chunk_size=400, overlap=50):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap

    return chunks

# Create embeddings and index
def create_vector_store(pdf_path, index_file="faiss_index.bin", chunks_file="chunks.npy"):
    texts = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(texts)

    np.save(chunks_file, chunks)
    print(f"Total chunks created: {len(chunks)}")
    print(f"Creating embeddings...")

    # Init Sentence Transformer Model
    model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    embeddings = model.encode(chunks)
    embeddings = np.array(embeddings).astype('float32')
    # Print embeddings size
    print(f"Embeddings size: {embeddings.shape}")
    print("Building index...")

    # Dimension size
    dimension = embeddings.shape[1]
    # Init index
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Save index to reuse
    faiss.write_index(index, index_file)
    print(f"Index saved to {index_file}")

    return index, chunks
