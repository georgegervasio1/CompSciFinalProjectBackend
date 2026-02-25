import os
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from openai import OpenAI

app = Flask(__name__)
CORS(app)

# Load OpenAI API key from environment variable
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)


# Health check route
@app.route("/")
def home():
    return "Backend is running."


# Proper CSV API route (BEST PRACTICE)
@app.route("/api/clients")
def get_clients():
    try:
        df = pd.read_csv("clients.csv")
        return jsonify(df.to_dict(orient="records"))
    except Exception as e:
        print("CSV ERROR:", str(e))
        return jsonify({"error": str(e)}), 500


# Optional: serve raw CSV if you still want it accessible
@app.route("/clients.csv")
def serve_csv():
    try:
        return send_from_directory(os.getcwd(), "clients.csv")
    except Exception as e:
        print("FILE ERROR:", str(e))
        return jsonify({"error": str(e)}), 500


# Chat endpoint
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

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful data analyst."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.5
        )

        answer = response.choices[0].message.content

        return jsonify({"response": answer})

    except Exception as e:
        print("CHAT ERROR:", str(e))
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
