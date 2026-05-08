import os
import csv
import io
import re
import random
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
def run_llama(prompt, max_tokens=400, temperature=0.3, extra_stop=None, allow_paragraphs=False):
    stop_sequences = ["Question:", "USER", "SYSTEM", "Will you assist", "Message End"]
    if not allow_paragraphs:
        stop_sequences.insert(0, "\n\n")
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


def classify_severity(c):
    """Classify a client's severity level based on key fields."""
    death = str(c.get("death", "")).strip().lower() in ("yes", "true", "1", "deceased", "death", "y")
    disability = str(c.get("disability", "")).strip().lower() in ("yes", "true", "1", "y")
    hospitalized = str(c.get("hospitalized", "")).strip().lower() in ("yes", "true", "1", "y")
    if death:
        return "catastrophic"
    elif disability:
        return "severe"
    elif hospitalized:
        return "moderate_high"
    else:
        return "moderate"


SEVERITY_RANGES = {
    "catastrophic":  (500_000, 2_000_000),
    "severe":        (150_000,   600_000),
    "moderate_high": ( 50_000,   200_000),
    "moderate":      ( 15_000,    75_000),
}

TIER_META = {
    "catastrophic":  ("TIER 1 — DEATH & CATASTROPHIC INJURY",        "$500,000 – $2,000,000+"),
    "severe":        ("TIER 2 — PERMANENT DISABILITY & SEVERE INJURY","$150,000 – $600,000"),
    "moderate_high": ("TIER 3 — HOSPITALIZATION & SIGNIFICANT INJURY","$50,000 – $200,000"),
    "moderate":      ("TIER 4 — DOCUMENTED MODERATE INJURY",          "$15,000 – $75,000"),
}

TIER_COLORS = {
    "catastrophic":  "#ef4444",
    "severe":        "#f97316",
    "moderate_high": "#eab308",
    "moderate":      "#22c55e",
}


def fmt_currency(amount):
    if amount >= 1_000_000:
        return f"${amount / 1_000_000:.1f}M"
    elif amount >= 1_000:
        return f"${amount / 1_000:.0f}K"
    return f"${amount:,.0f}"


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

        prompt = f"""You are a legal data analysis assistant. Use only projected, estimated, and forecasted language. Never say "final settlement", "guaranteed", or "will receive". Always say "projected", "estimated", or "forecasted".
When you are finished write: "Message End"
The dataset contains client injury cases with:
- Client ID, State, Incident Date, Injury Type, Product Brand

User question:
{user_message}

Answer clearly and concisely.
Do NOT show code. Do NOT use markdown. Return only a clean final answer as a normal paragraph.
ANSWER:
"""
        return jsonify({"response": run_llama(prompt, max_tokens=200)})

    except Exception as e:
        print("CHAT ERROR:", str(e))
        return jsonify({"error": str(e)}), 500


# ── Settlement Estimates ──────────────────────────────────────────────────
@app.route("/api/settlement-estimates", methods=["POST"])
def settlement_estimates():
    try:
        data = request.get_json()
        clients = data.get("clients", [])
        if not clients:
            return jsonify({"error": "No client data provided"}), 400

        # Step 1: Classify and assign estimates to every client
        classified = []
        for c in clients:
            severity = classify_severity(c)
            low, high = SEVERITY_RANGES[severity]
            # Use client ID as seed for reproducible estimates
            seed_val = hash(str(c.get("id", ""))) % 100_000
            rng = random.Random(seed_val)
            ratio = 0.30 + rng.random() * 0.50   # 30–80 % of range
            estimate_val = int(low + (high - low) * ratio)
            classified.append({**c, "severity": severity, "estimate_val": estimate_val})

        # Step 2: Generate LLM reasoning once per severity category
        reasoning_by_severity = {}
        for severity in SEVERITY_RANGES:
            group = [c for c in classified if c["severity"] == severity]
            if not group:
                continue

            sample_injuries = list({str(c.get("injury", ""))[:50] for c in group[:4]})
            low, high = SEVERITY_RANGES[severity]
            tier_name, _ = TIER_META[severity]

            severity_desc = {
                "catastrophic":  "death and catastrophic injury",
                "severe":        "permanent disability and severe long-term impairment",
                "moderate_high": "hospitalization and significant medical treatment",
                "moderate":      "documented injury requiring medical attention",
            }[severity]

            prompt = f"""You are a mass tort settlement analyst.
Write 2 sentences of projected settlement reasoning for clients with {severity_desc}.
Sample injury types: {', '.join(sample_injuries[:3])}.
Projected range: ${low:,} to ${high:,}.
Explain why these cases project in this range. Reference severity factors. Use only "projected" or "estimated" language. Do NOT state a dollar figure.
REASONING:"""

            reasoning_by_severity[severity] = run_llama(
                prompt, max_tokens=100, temperature=0.4,
                extra_stop=["Client:", "TIER", "Step"]
            ).strip()

        # Step 3: Build final per-client results
        result_clients = []
        for c in classified:
            severity = c["severity"]
            tier_name, est_range = TIER_META[severity]

            prefix = ""
            death_flag = str(c.get("death", "")).strip().lower() in ("yes", "true", "1", "deceased", "death", "y")
            disability_flag = str(c.get("disability", "")).strip().lower() in ("yes", "true", "1", "y")
            hosp_flag = str(c.get("hospitalized", "")).strip().lower() in ("yes", "true", "1", "y")

            if death_flag:
                prefix = "This case involves a fatality, qualifying for wrongful death compensation. "
            elif disability_flag:
                prefix = "Long-term disability documentation supports an elevated projected estimate. "
            elif hosp_flag:
                prefix = "Hospitalization records provide documented medical damages to support this projection. "

            base = reasoning_by_severity.get(severity, "Projected estimate based on documented injury severity and comparable mass tort outcomes.")
            result_clients.append({
                "id": c.get("id", "Unknown"),
                "injury": c.get("injury", "Not specified"),
                "state": c.get("state", "Unknown"),
                "estimate": f"${c['estimate_val']:,.0f}",
                "estimate_val": c["estimate_val"],
                "tier": tier_name,
                "severity_key": severity,
                "tier_color": TIER_COLORS[severity],
                "reasoning": prefix + base,
            })

        total_val = sum(c["estimate_val"] for c in classified)

        return jsonify({
            "total": fmt_currency(total_val),
            "total_raw": total_val,
            "client_count": len(result_clients),
            "clients": result_clients,
        })

    except Exception as e:
        print("SETTLEMENT ERROR:", str(e))
        return jsonify({"error": str(e)}), 500


# ── Litigation Costs ──────────────────────────────────────────────────────
@app.route("/api/litigation-costs", methods=["POST"])
def litigation_costs():
    try:
        data = request.get_json()
        clients = data.get("clients", [])
        client_count = len(clients) if clients else int(data.get("client_count", 0))
        if client_count == 0:
            return jsonify({"error": "No client data provided"}), 400

        # Rule-based cost projections
        attorney_count   = max(3, min(25, client_count // 40 + 3))
        hours_per_client = 14
        total_hours      = client_count * hours_per_client
        blended_rate     = 295

        attorney_cost    = total_hours * blended_rate
        expert_count     = max(3, min(15, client_count // 80 + 3))
        expert_unit_cost = 38_000
        expert_total     = expert_count * expert_unit_cost
        discovery_cost   = client_count * 1_350
        admin_cost       = client_count * 850
        filing_cost      = client_count * 275
        trial_prep_cost  = client_count * 2_200

        total_cost = (
            attorney_cost + expert_total + discovery_cost +
            admin_cost + filing_cost + trial_prep_cost
        )

        prompt = f"""You are a senior mass tort litigation cost analyst.
Write a 3-4 sentence professional narrative explaining the following projected litigation cost breakdown.

Portfolio: {client_count} clients across a mass tort matter.
Attorney Fees: {attorney_count} attorneys, {total_hours:,} projected hours at ${blended_rate}/hr blended rate.
Expert Witnesses: {expert_count} experts at ${expert_unit_cost:,} each.
Discovery & Document Review: ${discovery_cost:,} total.
Administrative & Case Management: ${admin_cost:,}.
Filing & Court Costs: ${filing_cost:,}.
Trial Preparation: ${trial_prep_cost:,}.
Total Projected Litigation Cost: ${total_cost:,.0f}.

Write analytically. Use only "projected" or "estimated" language. Explain the key cost drivers and how portfolio scale affects costs.
ANALYSIS:"""

        reasoning = run_llama(prompt, max_tokens=220, temperature=0.3, allow_paragraphs=True).strip()

        return jsonify({
            "total": fmt_currency(total_cost),
            "total_raw": total_cost,
            "breakdown": [
                {
                    "label": "Attorney Fees",
                    "detail": f"{attorney_count} attorneys × {total_hours:,} hrs @ ${blended_rate}/hr blended rate",
                    "amount": attorney_cost,
                    "formatted": fmt_currency(attorney_cost),
                },
                {
                    "label": "Expert Witnesses",
                    "detail": f"{expert_count} experts × ${expert_unit_cost:,} each",
                    "amount": expert_total,
                    "formatted": fmt_currency(expert_total),
                },
                {
                    "label": "Discovery & Document Review",
                    "detail": f"${1_350}/client × {client_count} clients",
                    "amount": discovery_cost,
                    "formatted": fmt_currency(discovery_cost),
                },
                {
                    "label": "Administrative & Case Management",
                    "detail": f"${850}/client × {client_count} clients",
                    "amount": admin_cost,
                    "formatted": fmt_currency(admin_cost),
                },
                {
                    "label": "Filing & Court Costs",
                    "detail": f"${275}/client × {client_count} clients",
                    "amount": filing_cost,
                    "formatted": fmt_currency(filing_cost),
                },
                {
                    "label": "Trial Preparation & Litigation Support",
                    "detail": f"${2_200}/client × {client_count} clients",
                    "amount": trial_prep_cost,
                    "formatted": fmt_currency(trial_prep_cost),
                },
            ],
            "reasoning": reasoning,
            "attorney_count": attorney_count,
            "total_hours": total_hours,
            "blended_rate": blended_rate,
            "expert_count": expert_count,
        })

    except Exception as e:
        print("LITIGATION COST ERROR:", str(e))
        return jsonify({"error": str(e)}), 500


# ── Tier Analysis (Automatic) ─────────────────────────────────────────────
@app.route("/api/tier-analysis", methods=["POST"])
def tier_analysis():
    try:
        data = request.get_json()
        clients = data.get("clients", [])
        if not clients:
            return jsonify({"error": "No client data provided"}), 400

        # Classify all clients
        groups = {k: [] for k in SEVERITY_RANGES}
        for c in clients:
            groups[classify_severity(c)].append(c)

        tier_results = []
        for severity_key in ("catastrophic", "severe", "moderate_high", "moderate"):
            group = groups[severity_key]
            if not group:
                continue

            tier_name, est_range = TIER_META[severity_key]
            tier_color = TIER_COLORS[severity_key]
            sample_injuries = list({str(c.get("injury", "Unknown"))[:50] for c in group[:5]})
            all_ids = [str(c.get("id", "Unknown")) for c in group]

            severity_desc = {
                "catastrophic":  "death and catastrophic injury",
                "severe":        "permanent disability and severe long-term injury",
                "moderate_high": "hospitalization and documented significant injury",
                "moderate":      "moderate injury with medical documentation",
            }[severity_key]

            prompt = f"""You are a senior mass tort litigation analyst.
Write 3 sentences classifying clients in the {tier_name} tier.

Clients in this tier: {len(group)} (out of {len(clients)} total)
Severity characteristics: {severity_desc}
Sample injury types observed: {', '.join(sample_injuries[:4])}
Projected settlement range: {est_range}

Sentence 1: Describe what injury or harm characteristics define this tier.
Sentence 2: Explain why these characteristics place clients in this projected settlement range.
Sentence 3: Note any key factors that could increase or decrease individual estimates within the tier.
Use only projected or estimated language. Do not use markdown or bullet points.
ANALYSIS:"""

            reasoning = run_llama(
                prompt, max_tokens=160, temperature=0.35,
                allow_paragraphs=True,
                extra_stop=["Sentence", "TIER", "Tier 1", "Tier 2", "Tier 3", "Tier 4"]
            ).strip()

            tier_results.append({
                "severity_key": severity_key,
                "tier_name": tier_name,
                "estimate_range": est_range,
                "tier_color": tier_color,
                "client_count": len(group),
                "all_client_ids": all_ids,
                "sample_client_ids": all_ids[:10],
                "reasoning": reasoning,
            })

        return jsonify({
            "tiers": tier_results,
            "total_clients": len(clients),
            "tier_count": len(tier_results),
        })

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

        prompt = f"""You are a prompt engineer specializing in legal data analysis queries.
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
        file        = request.files.get("file")
        matter_name = request.form.get("matter_name", "Default").strip()
        is_new      = request.form.get("is_new", "true").lower() == "true"

        if not file:
            return jsonify({"error": "No file provided"}), 400
        if not matter_name:
            return jsonify({"error": "No matter name provided"}), 400

        content = file.read().decode("utf-8-sig")
        reader  = csv.DictReader(io.StringIO(content))
        records = []
        for row in reader:
            cleaned = {
                k.strip(): (v.strip() if v and v.strip() != "NaN" else None)
                for k, v in row.items()
            }
            records.append(cleaned)

        if not records:
            return jsonify({"error": "CSV is empty or could not be parsed"}), 400

        if is_new or matter_name not in matters:
            matters[matter_name] = records
        else:
            matters[matter_name].extend(records)

        return jsonify({
            "success": True,
            "matter":  matter_name,
            "count":   len(matters[matter_name]),
            "records": matters[matter_name],
        })

    except Exception as e:
        print("UPLOAD ERROR:", str(e))
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
