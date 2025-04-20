import numpy as np
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config import API_KEY
import google.generativeai as genai

genai.configure(api_key=API_KEY)
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def find_relevant_images_by_embedding(query, image_embeddings, top_k=1):
    query_embedding = embedding_model.embed_query(query)
    scores = []
    for page, data in image_embeddings.items():
        score = cosine_similarity(query_embedding, data["embedding"])
        scores.append((page, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    top_pages = [p for p, _ in scores[:top_k]]
    return [image_embeddings[p]["image"] for p in top_pages]
