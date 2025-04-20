import os
from dotenv import load_dotenv

load_dotenv()

POPLER_PATH = r"C:\Users\KARTHICK\poppler-24.08.0\Library\bin"
POLICY_PDF_PATH = "data/Company-Policy-and-Procedure-June-1.18-V6.0.pdf"
IMAGE_FOLDER = "data/saved_flowchart_images"
FAISS_INDEX_PATH = "data/faiss_index"
IMAGE_EMBED_FILE = "data/image_embeddings.pkl"
FLOWCHART_PAGES = [9, 10, 11, 21, 25, 32, 51]

API_KEY = os.getenv("MY_API_KEY")
GOOGLE_CREDENTIALS = "service-account-key.json"