import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import tinker

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

API_KEY = os.environ.get("TINKER_API_KEY")

service_client = tinker.ServiceClient(api_key=API_KEY)
model_name = "meta-llama/Llama-3.1-8B-Instruct"
client = service_client.create_sampling_client(base_model=model_name)
tokenizer = client.get_tokenizer()


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

        # SMALL, STATIC CONTEXT (no pandas, no CSV)
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

        tokens = tokenizer.encode(prompt)
        model_input = tinker.types.ModelInput.from_ints(tokens)

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
    app.run(host="0.0.0.0", port=5000)
