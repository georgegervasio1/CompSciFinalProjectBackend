import os
import csv
import io
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import tinker

app = Flask(__name__)
CORS(app)

API_KEY = os.environ.get("TINKER_API_KEY")
service_client = tinker.ServiceClient(api_key=API_KEY)
model_name = "meta-llama/Llama-3.1-8B-Instruct"
client = service_client.create_sampling_client(base_model=model_name)
tokenizer = client.get_tokenizer()

# In-memory matter storage: { matter_name: [records] }
matters = {}


# ── Helpers ──────────────────────────────────────────────────────────────
def run_llama(prompt, max_tokens=400, temperature=0.3, extra_stop=None):
    stop_sequences = ["\n\n", "Question:", "USER", "SYSTEM", "Will you assist", "Message End"]
    if extra_stop:
        stop_sequences.extend(extra_stop)
    tokens = tokenizer.encode(prompt)
    model_input = tinker.types.ModelInput.from_ints(tokens)
    result = client.sample(
        prompt=model_input,
        num_samples=1,
        sampling_params=tinker.types.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop_sequences
        )
    ).result()
    return tokenizer.decode(result.sequences[0].tokens)


# ── Routes ───────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return "Backend is running."


@app.route("/clients.csv")
def get_clients():
    try:
        return send_file("clients.csv")
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Chat ─────────────────────────────────────────────────────────────────
@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_message = data.get("message", "")
        if not user_message:
            return jsonify({"error": "No message provided"}), 400

        prompt = f"""
You are a legal data analysis assistant. Use only projected, estimated, and forecasted language. Never say "final settlement", "guaranteed", or "will receive". Always say "projected", "estimated", or "forecasted".
When you are finished write: "Message End"
The dataset contains client injury cases with:
- Client ID
- State
- Incident Date
- Injury Type
- Product Brand

User question:
{user_message}

Answer the question clearly and concisely.
Do NOT show code.
Do NOT show methodology.
Do NOT use markdown.
Do NOT explain your reasoning.
Return only a clean final answer written as a normal paragraph.
Expand only when necessary.
ANSWER:
"""
        return jsonify({"response": run_llama(prompt, max_tokens=200)})

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({"error": str(e)}), 500


# ── Tier Analysis ─────────────────────────────────────────────────────────
@app.route("/api/tier-analysis", methods=["POST"])
def tier_analysis():
    try:
        data = request.get_json()
        clients = data.get("clients", [])
        tier_count = int(data.get("tier_count", 3))

        if not clients:
            return jsonify({"error": "No client data provided"}), 400

        # Build severity summary for prompt
        total = len(clients)
        death_count = sum(1 for c in clients if str(c.get("death", "")).strip().lower() in ("yes", "true", "1", "deceased", "death"))
        hosp_count = sum(1 for c in clients if str(c.get("hospitalized", "")).strip().lower() in ("yes", "true", "1"))
        disability_count = sum(1 for c in clients if str(c.get("disability", "")).strip().lower() in ("yes", "true", "1"))

        injury_counts = {}
        for c in clients:
            inj = str(c.get("injury", "Unknown")).strip()[:60]
            injury_counts[inj] = injury_counts.get(inj, 0) + 1

        top_injuries = sorted(injury_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        injury_summary = "; ".join(f"{k} ({v})" for k, v in top_injuries)

        state_counts = {}
        for c in clients:
            s = str(c.get("state", "Unknown")).strip()
            state_counts[s] = state_counts.get(s, 0) + 1
        top_states = sorted(state_counts.items(), key=lambda x: x[1], reverse=True)[:6]
        state_summary = "; ".join(f"{k} ({v})" for k, v in top_states)

        sample_ids = [str(c.get("id", "")) for c in clients[:12] if c.get("id")]

        tier_label_map = {
            3: "Tier 1 (Highest Projected Value), Tier 2 (Medium Projected Value), Tier 3 (Lower Projected Value)",
            5: "Tier 1 (Highest), Tier 2 (High), Tier 3 (Medium), Tier 4 (Low), Tier 5 (Lowest)"
        }
        tier_labels = tier_label_map.get(tier_count, " ".join(f"Tier {i}" for i in range(1, tier_count + 1)))

        prompt = f"""
You are a senior mass tort litigation analyst. Your job is to group {total} clients into {tier_count} projected settlement tiers based on case severity.

Dataset Summary:
- Total Clients: {total}
- Deaths: {death_count}
- Hospitalizations: {hosp_count}
- Long-term Disability: {disability_count}
- Top Injury Types: {injury_summary}
- Top States: {state_summary}
- Sample Client IDs: {", ".join(sample_ids)}

Tier Structure: {tier_labels}

Instructions:
Group clients into exactly {tier_count} tiers. For each tier write: the tier name, the estimated number of clients in that tier, a 2-3 sentence summary of the severity characteristics that place clients in this tier, and example client ID types or characteristics. Base groupings on death, hospitalization, long-term disability, and injury severity. Use only projected or estimated language. Never say guaranteed or final.
Do NOT use markdown. Do NOT use bullet points. Write each tier as a plain paragraph starting with the tier name in all caps.
ANALYSIS:
"""
        response_text = run_llama(prompt, max_tokens=500, temperature=0.4)
        return jsonify({"response": response_text})

    except Exception as e:
        print("TIER ERROR:", str(e))
        return jsonify({"error": str(e)}), 500


# ── Refine Prompt ─────────────────────────────────────────────────────────
@app.route("/api/refine-prompt", methods=["POST"])
def refine_prompt():
    try:
        data = request.get_json()
        user_prompt = data.get("prompt", "").strip()

        if not user_prompt:
            return jsonify({"error": "No prompt provided"}), 400

        prompt = f"""
You are a prompt engineer specializing in legal data analysis queries.
A user typed the following question to ask an AI about a mass tort client dataset:

Original: "{user_prompt}"

Rewrite this into a single, improved question that is more specific, data-driven, and likely to produce a useful analytical answer about injury patterns, projected settlement values, geographic trends, or case characteristics.
Return ONLY the improved question. No explanation. No preamble. No quotes. Just the question.
IMPROVED QUESTION:
"""
        refined = run_llama(prompt, max_tokens=80, temperature=0.4,
                            extra_stop=["Original:", "User:", "\n"])
        refined = refined.strip().strip('"').strip("'")
        return jsonify({"refined": refined})

    except Exception as e:
        print("REFINE ERROR:", str(e))
        return jsonify({"error": str(e)}), 500


# ── Upload CSV ────────────────────────────────────────────────────────────
@app.route("/api/upload-csv", methods=["POST"])
def upload_csv():
    try:
        file = request.files.get("file")
        matter_name = request.form.get("matter_name", "Default").strip()
        is_new = request.form.get("is_new", "true").lower() == "true"

        if not file:
            return jsonify({"error": "No file provided"}), 400
        if not matter_name:
            return jsonify({"error": "No matter name provided"}), 400

        content = file.read().decode("utf-8-sig")  # utf-8-sig handles BOM
        reader = csv.DictReader(io.StringIO(content))
        records = []
        for row in reader:
            cleaned = {k.strip(): (v.strip() if v and v.strip() != "NaN" else None)
                       for k, v in row.items()}
            records.append(cleaned)

        if not records:
            return jsonify({"error": "CSV is empty or could not be parsed"}), 400

        if is_new or matter_name not in matters:
            matters[matter_name] = records
        else:
            matters[matter_name].extend(records)

        return jsonify({
            "success": True,
            "matter": matter_name,
            "count": len(matters[matter_name]),
            "records": matters[matter_name]
        })

    except Exception as e:
        print("UPLOAD ERROR:", str(e))
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
