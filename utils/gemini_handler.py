import google.generativeai as genai
from config import API_KEY

genai.configure(api_key=API_KEY)

def describe_image_with_gemini(image):
    print(" Describing image using Gemini...")

    model = genai.GenerativeModel("gemini-1.5-pro-002")
    response = model.generate_content(["Describe this flowchart image in detail for semantic understanding.", image])
    print(" Image description generated.")

    return response.text.strip()

def generate_response(prompt, images=None):
    print(f" Finding relevant response for query:")

    print(" Generating response using Gemini with HTML formatting...")

    model = genai.GenerativeModel("gemini-1.5-pro-002")
    contents = [
        "You are an assistant helping users based on PDF documents and flowchart images.",
        "Please answer in a structured HTML format using elements like <h3>, <ul>, <ol>, <li>, <b>, and <br> for better readability.",
        "Do not wrap everything in a <html> or <body> tag. Just provide clean inner HTML.",
        prompt
    ]
    if images:
        contents += images
    response = model.generate_content(contents)
    print(" HTML-formatted response generated from Gemini.")

    return response.text.strip()
