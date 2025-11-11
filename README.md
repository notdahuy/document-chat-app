RAG Chatbot Document Q&A System
This is a Retrieval-Augmented Generation (RAG) system designed to allow users to upload PDF or DOCX documents and ask questions based on their content. 
The application provides fast, context-aware answers, demonstrating proficiency in building an end-to-end LLM solution.

Key Features
Multilingual Interface (i18n): Users can switch the UI language between Vietnamese and English.
Automatic Language Detection: The system automatically detects the language of the user's question (e.g., English, Vietnamese) and responds in the same language.
Cross-Lingual QA: Capable of finding and using context written in one language to answer a question asked in another (e.g., English question answered from a Vietnamese document).
Multi-Format File Support: Supports uploading and processing PDF and DOCX files.
Core Technologies: Utilizes FAISS for vector indexing and Google Gemini 2.5 Flash for answer generation.
