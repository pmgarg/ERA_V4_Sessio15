import os
import torch
import tiktoken
from flask import Flask, request, jsonify, render_template_string
from model import DeepSeekV3Config, DeepSeekV3ForCausalLM

app = Flask(__name__)

# -----------------------------------------------------------------------------
# Model Loading
# -----------------------------------------------------------------------------
MODEL_PATH = "checkpoint_5500.pt"  # Ensure this matches your final checkpoint name
DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"

print(f"Loading model from {MODEL_PATH} on {DEVICE}...")

# Initialize model structure
config = DeepSeekV3Config(
    vocab_size=50257,
    hidden_size=576,
    num_hidden_layers=30,
    num_attention_heads=9,
    kv_lora_rank=512,
    moe_intermediate_size=256,
    n_shared_experts=1,
    n_routed_experts=8,
    num_experts_per_tok=2,
    max_position_embeddings=2048,
)
model = DeepSeekV3ForCausalLM(config)

# Load weights if available
if os.path.exists(MODEL_PATH):
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    # Handle both full checkpoint dict and state_dict only
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=False)
    print("✅ Model weights loaded successfully!")
else:
    print(f"⚠️ Warning: Checkpoint {MODEL_PATH} not found. Using random weights.")

model.to(DEVICE)
model.eval()

# Tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

# -----------------------------------------------------------------------------
# HTML Template
# -----------------------------------------------------------------------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DeepSeek-V3 Text Generator</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; background-color: #f5f5f5; }
        .container { background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; text-align: center; }
        .input-group { margin-bottom: 20px; }
        label { display: block; margin-bottom: 8px; font-weight: bold; color: #34495e; }
        textarea { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 5px; min-height: 100px; font-size: 16px; }
        button { background-color: #3498db; color: white; border: none; padding: 12px 24px; border-radius: 5px; cursor: pointer; font-size: 16px; transition: background 0.3s; }
        button:hover { background-color: #2980b9; }
        button:disabled { background-color: #bdc3c7; cursor: not-allowed; }
        #output { margin-top: 20px; padding: 15px; border: 1px solid #e0e0e0; border-radius: 5px; background-color: #f9f9f9; min-height: 50px; white-space: pre-wrap; }
        .examples { margin-top: 30px; }
        .example-btn { background-color: #ecf0f1; color: #2c3e50; margin-right: 10px; margin-bottom: 10px; font-size: 14px; padding: 8px 15px; }
        .example-btn:hover { background-color: #d5dbdb; }
        .loader { display: none; border: 4px solid #f3f3f3; border-top: 4px solid #3498db; border-radius: 50%; width: 20px; height: 20px; animation: spin 1s linear infinite; display: inline-block; vertical-align: middle; margin-left: 10px; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <div class="container">
        <h1>DeepSeek-V3 Inference</h1>
        
        <div class="input-group">
            <label for="prompt">Enter your prompt:</label>
            <textarea id="prompt" placeholder="Once upon a time..."></textarea>
        </div>
        
        <div class="input-group">
            <label>Max Tokens: <span id="token-count">50</span></label>
            <input type="range" id="max-tokens" min="10" max="200" value="50" oninput="document.getElementById('token-count').innerText = this.value">
        </div>

        <button onclick="generate()" id="gen-btn">Generate Text</button>
        <div id="loader" class="loader" style="display: none;"></div>

        <div id="output"></div>

        <div class="examples">
            <h3>Try Examples:</h3>
            <button class="example-btn" onclick="setPrompt('The meaning of life is')">Meaning of Life</button>
            <button class="example-btn" onclick="setPrompt('In a galaxy far far away')">Sci-Fi</button>
            <button class="example-btn" onclick="setPrompt('Python is a programming language that')">Coding</button>
        </div>
    </div>

    <script>
        function setPrompt(text) {
            document.getElementById('prompt').value = text;
        }

        async function generate() {
            const prompt = document.getElementById('prompt').value;
            const maxTokens = document.getElementById('max-tokens').value;
            const btn = document.getElementById('gen-btn');
            const loader = document.getElementById('loader');
            const output = document.getElementById('output');

            if (!prompt) return;

            btn.disabled = true;
            loader.style.display = 'inline-block';
            output.innerText = 'Generating...';

            try {
                const response = await fetch('/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prompt: prompt, max_new_tokens: parseInt(maxTokens) })
                });
                
                const data = await response.json();
                if (data.error) {
                    output.innerText = 'Error: ' + data.error;
                } else {
                    output.innerText = data.text;
                }
            } catch (e) {
                output.innerText = 'Error: ' + e.message;
            } finally {
                btn.disabled = false;
                loader.style.display = 'none';
            }
        }
    </script>
</body>
</html>
"""

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get('prompt', '')
    max_new_tokens = data.get('max_new_tokens', 50)
    temperature = data.get('temperature', 0.8)
    top_k = data.get('top_k', 50)

    if not prompt:
        return jsonify({'error': 'Prompt is required'}), 400

    try:
        # Encode
        input_ids = tokenizer.encode(prompt)
        input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)

        # Generate
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_tensor,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k
            )

        # Decode
        generated_text = tokenizer.decode(output_ids[0].cpu().tolist())
        
        return jsonify({'text': generated_text})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)
