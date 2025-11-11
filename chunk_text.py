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


if __name__ == "__main__":
    from read_pdf import extract_text_from_pdf

    text = extract_text_from_pdf("open_circuits.pdf")
    chunks = chunk_text(text)

    print(f"Total chunks: {len(chunks)}\n")
    print("Preview chunk")
    print(chunks[0])