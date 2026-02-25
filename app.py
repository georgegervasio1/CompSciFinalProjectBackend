import os
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load API key from environment variable
API_KEY = os.environ.get("TINKER_API_KEY")

# Initialize Tinker client
service_client = tinker.ServiceClient(api_key=API_KEY)

model_name = "meta-llama/Llama-3.1-8B-Instruct"
client = service_client.create_sampling_client(base_model=model_name)
tokenizer = client.get_tokenizer()


# Root health check
@app.route("/")
def home():
    return "Backend is running."


# Serve clients.csv
@app.route("/clients.csv")
def get_clients():
    try:
        return send_file("clients.csv")
    except Exception as e:
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

        # Encode prompt
        tokens = tokenizer.encode(prompt)
        model_input = tinker.types.ModelInput.from_ints(tokens)

        # Call model (EXACT working pattern from class)
        result = client.sample(
            prompt=model_input,
            num_samples=1,
            sampling_params=tinker.types.SamplingParams(
                max_tokens=200,
                temperature=0.5
            )
        ).result()

        generated_tokens = result.sequences[0].tokens
        generated_text = tokenizer.decode(generated_tokens)

        return jsonify({"response": generated_text})

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
