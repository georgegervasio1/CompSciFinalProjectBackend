import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

API_KEY = os.environ.get("TINKER_API_KEY")

# CHANGE THIS to your real endpoint if different
LLM_ENDPOINT = "https://api.together.xyz/v1/completions"

@app.route("/")
def home():
    return "Backend is running."


@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_message = data.get("message", "")

        if not user_message:
            return jsonify({"error": "No message provided"}), 400

        prompt = f"""
You are a data analysis assistant.

The dataset contains client injury cases with:
- Client ID
- State
- Incident Date
- Injury Type
- Product Brand

User question:
{user_message}

Provide a concise analytical answer.
"""

        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "meta-llama/Llama-3.1-8B-Instruct",
            "prompt": prompt,
            "max_tokens": 200,
            "temperature": 0.5
        }

        response = requests.post(LLM_ENDPOINT, json=payload, headers=headers)

        result = response.json()

        generated_text = result["choices"][0]["text"]

        return jsonify({"response": generated_text})

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
