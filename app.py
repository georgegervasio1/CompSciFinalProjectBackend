from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import tinker
import pandas as pd

app = Flask(__name__)
CORS(app)

# This is my tinker API key
API_KEY = "tml-MjEL7v0bwky6oIqf65BdZvnPR6ovRLI494ktF5odhq8vEFNPAzLfBO6lOq42CfAqFAAAA"

# Initialize service client with API key
service_client = tinker.ServiceClient(api_key=API_KEY)
model_name = "meta-llama/Llama-3.1-8B-Instruct"

client = service_client.create_sampling_client(base_model=model_name)
tokenizer = client.get_tokenizer()

# Load client data for context
def load_csv_summary():
    """Load and summarize CSV data for LLM context"""
    try:
        df = pd.read_csv('clients.csv')
        summary = f"""
Dataset Summary:
- Total clients: {len(df)}
- States represented: {df['Current Residence State'].nunique() if 'Current Residence State' in df.columns else 'N/A'}
- Date range: {df['Incident Date'].min() if 'Incident Date' in df.columns else 'N/A'} to {df['Incident Date'].max() if 'Incident Date' in df.columns else 'N/A'}

Available columns: {', '.join(df.columns.tolist())}
"""
        return summary, df
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None, None

data_summary, df = load_csv_summary()

@app.route('/')
def serve_html():
    """Serve the main HTML file"""
    return send_from_directory('.', 'index.html')

@app.route('/clients.csv')
def serve_csv():
    """Serve the CSV file"""
    return send_from_directory('.', 'clients.csv')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat requests - connects to Tinker LLM"""
    try:
        data = request.json
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Build the prompt with context
        system_prompt = """You are a data analysis assistant for client injury data. Answer questions about the data concisely and accurately.

"""
        if data_summary:
            system_prompt += data_summary + "\n\n"
        
        # THIS IS WHERE YOUR PYTHON CODE RUNS
        prompt = system_prompt + f"User question: {user_message}\n\nAnswer:"
        
        # Tokenize and create model input
        tokens = tokenizer.encode(prompt)
        model_input = tinker.types.ModelInput.from_ints(tokens)
        
        # Sample from model (your exact code)
        result = client.sample(
            prompt=model_input,
            num_samples=1,
            sampling_params=tinker.types.SamplingParams(
                max_tokens=200,  # Increased for better answers
                temperature=0.5
            )
        ).result()
        
        # Decode the response
        generated_tokens = result.sequences[0].tokens
        generated_text = tokenizer.decode(generated_tokens)
        
        # Clean up the response (remove the prompt echo)
        response_text = generated_text.replace(prompt, '').strip()
        
        print(f"User: {user_message}")
        print(f"AI: {response_text}\n")
        
        return jsonify({'response': response_text})
        
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Flask server...")
    print("📊 Loading client data...")
    print("🤖 Tinker LLM ready")
    print("\n💻 Open http://localhost:5000 in your browser\n")
    app.run(debug=True, port=5000)
