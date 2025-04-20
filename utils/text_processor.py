import os
from pathlib import Path
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config import POLICY_PDF_PATH, FAISS_INDEX_PATH
from config import API_KEY
import google.generativeai as genai

genai.configure(api_key=API_KEY)
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

def load_pdf_text_with_page_info(pdf_path):
    pdf_reader = PdfReader(pdf_path)
    chunks = []
    for i, page in enumerate(pdf_reader.pages):
        text = page.extract_text()
        if text:
            splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
            for chunk in splitter.split_text(text):
                chunks.append((chunk, {"page": i + 1}))
    print(" PDF text extraction completed.")

    return chunks

def create_vector_store(chunks_with_meta):
    print(" Creating vector store from extracted PDF text...")

    texts = [chunk for chunk, _ in chunks_with_meta]
    metadatas = [meta for _, meta in chunks_with_meta]
    return FAISS.from_texts(texts=texts, embedding=embedding_model, metadatas=metadatas)

def init_vectorstore():
    if not os.path.exists(FAISS_INDEX_PATH):
        chunks = load_pdf_text_with_page_info(POLICY_PDF_PATH)
        vectorstore = create_vector_store(chunks)
        vectorstore.save_local(FAISS_INDEX_PATH)
    else:
        vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings=embedding_model, allow_dangerous_deserialization=True)
    return vectorstore
