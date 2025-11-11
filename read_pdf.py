from pypdf import PdfReader

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    all_text = ""

    for page in reader.pages:
        text = page.extract_text()
        if text:
            all_text += text + "\n"

    return all_text


if __name__ == "__main__":
    pdf_path = "open_circuits.pdf"
    text = extract_text_from_pdf(pdf_path)

    print("PDF Preview")
    print(text[:2000])