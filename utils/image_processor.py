from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image
import pickle
from config import FLOWCHART_PAGES, POPLER_PATH, POLICY_PDF_PATH, IMAGE_FOLDER, IMAGE_EMBED_FILE
from utils.gemini_handler import describe_image_with_gemini
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config import API_KEY
import google.generativeai as genai

genai.configure(api_key=API_KEY)
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

def extract_and_save_images():
    print(" Extracting and saving flowchart images...")

    Path(IMAGE_FOLDER).mkdir(parents=True, exist_ok=True)
    for page in FLOWCHART_PAGES:
        img = convert_from_path(POLICY_PDF_PATH, poppler_path=POPLER_PATH, first_page=page, last_page=page)[0]
        img_path = Path(IMAGE_FOLDER) / f"page_{page}.jpg"
        img.save(img_path, "JPEG", quality=70)


def build_image_embeddings():
    print(" Generating image embeddings for flowcharts...")

    image_embeddings = {}
    for page in FLOWCHART_PAGES:
        img_path = Path(IMAGE_FOLDER) / f"page_{page}.jpg"
        if not img_path.exists():
            continue
        img = Image.open(img_path)
        desc = describe_image_with_gemini(img)
        embedding = embedding_model.embed_query(desc)
        image_embeddings[page] = {
            "image": img,
            "desc": desc,
            "embedding": embedding
        }
    with open(IMAGE_EMBED_FILE, "wb") as f:
        pickle.dump(image_embeddings, f)
    print(" Image embeddings saved to file.")

    return image_embeddings

def load_image_embeddings():
    if Path(IMAGE_EMBED_FILE).exists():
        print(" Loading pre-generated image embeddings...")

        with open(IMAGE_EMBED_FILE, "rb") as f:
            return pickle.load(f)
    return build_image_embeddings()

def prepare_images_and_embeddings():
    if not Path(IMAGE_FOLDER).exists() or not list(Path(IMAGE_FOLDER).glob("*.jpg")):
        extract_and_save_images()
    return load_image_embeddings()
