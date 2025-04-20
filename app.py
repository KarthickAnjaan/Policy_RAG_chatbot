from flask import Flask, request, jsonify
from flask_cors import CORS
from utils.text_processor import init_vectorstore
from utils.image_processor import prepare_images_and_embeddings
from utils.similarity import find_relevant_images_by_embedding
from utils.gemini_handler import generate_response

app = Flask(__name__)
CORS(app)

vectorstore = init_vectorstore()
image_embeddings = prepare_images_and_embeddings()

@app.route("/query", methods=["POST"])
def query_ltn():
    data = request.get_json()
    query = data.get("query", "")
    if not query:
        return jsonify({"error": "Query is required"}), 400

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    relevant_docs = retriever.invoke(query)
    combined_text = "\n".join([doc.page_content for doc in relevant_docs])
    selected_images = find_relevant_images_by_embedding(query, image_embeddings, top_k=1)

    prompt = f"Based on the following extracted text:\n\n{combined_text}\n\n"
    if selected_images:
        prompt += f"Also analyze the flowchart image(s). Now answer:\n\n{query}"
    else:
        prompt += f"Now answer:\n\n{query}"

    response = generate_response(prompt, images=selected_images)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)