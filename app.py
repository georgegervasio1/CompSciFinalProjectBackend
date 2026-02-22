import os
import pandas as pd
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import tinker

app = Flask(__name__)
CORS(app)

# Load API key securely from Render environment variables
API_KEY = os.environ.get("TINKER_API_KEY")

# Initialize Tinker client EXACTLY like your class example
service_client = tinker.ServiceClient(api_key=API_KEY)

model_name = "meta-llama/Llama-3.1-8B-Instruct"
client = service_client.create_sampling_client(base_model=model_name)
tokenizer = client.get_tokenizer()

# Load CSV once at startup
CSV_PATH = "clients.csv"
if os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH)
else:
    df = None


# Health check route
@app.route("/")
def home():
    return "Backend is running."


# Serve CSV to frontend
@app.route("/clients.csv")
def get_csv():
    return send_file(CSV_PATH)


# Chat endpoint
@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_message = data.get("message", "")

        if not user_message:
            return jsonify({"error": "No message provided"}), 400

        # Convert dataframe to string (limit size)
        if df is not None:
            data_preview = df.head(200).to_string()
        else:
            data_preview = "No client data loaded."

        prompt = f"""
You are a data analysis assistant.

Here is the client dataset:
{data_preview}

User question:
{user_message}

Provide a clear and concise answer based only on the dataset.
"""

        # ----- TINKER LOGIC (MATCHES YOUR WORKING CLASS CODE) -----

        tokens = tokenizer.encode(prompt)
        model_input = tinker.types.ModelInput.from_ints(tokens)

        result = client.sample(
            prompt=model_input,
            num_samples=1,
            sampling_params=tinker.types.SamplingParams(
                max_tokens=300,
                temperature=0.7
            )
        ).result()

        generated_tokens = result.sequences[0].tokens
        generated_text = tokenizer.decode(generated_tokens)

        return jsonify({"response": generated_text})

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
