# Policy_RAG_chatbot

# 📄 LTN Assistant – PDF & Flowchart QA System

This project is an intelligent assistant built with **Flask**, powered by **Google Gemini 1.5 Pro**. It allows users to query information from a PDF document along with supporting flowchart images.

It uses:
- PDF parsing
- Flowchart image understanding
- Text & image embeddings
- Gemini 1.5 Pro multimodal model
- FAISS vector search for document chunk retrieval

---

## 🚀 Features

✅ Extracts meaningful content from PDF documents  
✅ Converts selected pages into flowchart images  
✅ Uses Gemini 1.5 Pro for smart, structured responses  
✅ Combines image and text input for multimodal queries  
✅ Returns clean, HTML-formatted answers

---

## ⚙️ Requirements

### ✅ 1. Python Environment


Install the required libraries:

```bash
pip install -r requirements.txt
```



### 2.  Google Gemini Access
To use Gemini 1.5 Pro:

Get a Google Gemini API key from Google AI Studio.

Create a service account key JSON file from Google Cloud Console.

### 3.Poppler (for PDF to Image conversion)
Install Poppler to convert PDF pages into images:

Download Poppler for Windows
Extract it, and set the POPLER_PATH in your code to the bin folder.

Example
```bash
POPLER_PATH = r"C:\Users\YourName\poppler-xx\Library\bin"

```
