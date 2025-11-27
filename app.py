"""
SmolLM2-135M Shakespeare Generator
Hugging Face Gradio App for trained model deployment
"""

import gradio as gr
import torch
import tiktoken
from model import SmolLM2Config, SmolLM2ForCausalLM
import os

# ============================================================================
# MODEL LOADING
# ============================================================================

print("Loading SmolLM2-135M Shakespeare Model...")

# Device setup
if torch.cuda.is_available():
    device = 'cuda'
    print(f"Using device: CUDA ({torch.cuda.get_device_name(0)})")
elif torch.backends.mps.is_available():
    device = 'mps'
    print(f"Using device: MPS (Apple Silicon)")
else:
    device = 'cpu'
    print(f"Using device: CPU")

# Load checkpoint
checkpoint_path = "checkpoint_5500.pt"
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
config = checkpoint['config']
model = SmolLM2ForCausalLM(config)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

print(f"✓ Model loaded successfully!")
print(f"  Parameters: {checkpoint['total_params']:,}")
print(f"  Training steps: {checkpoint['global_step']:,}")

# Load tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

# ============================================================================
# GENERATION FUNCTION
# ============================================================================

def generate_shakespeare(
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 50
) -> str:
    """
    Generate Shakespeare-style text from a prompt.

    Args:
        prompt: Starting text
        max_tokens: Number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Top-k filtering (smaller = more focused)

    Returns:
        Generated text
    """
    if not prompt.strip():
        return "Please enter a prompt!"

    try:
        # Encode prompt
        input_ids = tokenizer.encode(prompt)
        input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)

        # Generate
        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k
            )

        # Decode
        generated_text = tokenizer.decode(generated[0].cpu().tolist())
        return generated_text

    except Exception as e:
        return f"Error during generation: {str(e)}"

# ============================================================================
# GRADIO INTERFACE
# ============================================================================

# Example prompts
examples = [
    ["To be or not to be", 100, 0.8, 50],
    ["Once upon a time in fair Verona", 150, 0.7, 40],
    ["Friends, Romans, countrymen", 120, 0.8, 50],
    ["Now is the winter of our discontent", 100, 0.9, 60],
    ["The quality of mercy is not strained", 80, 0.7, 40],
]

# Custom CSS for better styling
css = """
#title {
    text-align: center;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.5em;
    font-weight: bold;
    margin-bottom: 0.5em;
}
#description {
    text-align: center;
    font-size: 1.1em;
    color: #666;
    margin-bottom: 2em;
}
.metrics {
    background: #f0f0f0;
    padding: 15px;
    border-radius: 10px;
    margin: 10px 0;
}
"""

# Create Gradio interface
with gr.Blocks(css=css, theme=gr.themes.Soft()) as demo:
    # Header
    gr.Markdown("<h1 id='title'>🎭 SmolLM2-135M Shakespeare Generator</h1>")
    gr.Markdown(
        "<p id='description'>Generate Shakespeare-style text using a custom-trained 135M parameter language model</p>"
    )

    # Model Info
    with gr.Accordion("📊 Model Information", open=False):
        gr.Markdown(f"""
        ### Training Details
        - **Architecture**: SmolLM2-135M (LLaMA-style)
        - **Parameters**: 135,151,488 (135M)
        - **Training Steps**: 5,500
        - **Final Loss**: 0.0530
        - **Perplexity**: 1.054
        - **Training Time**: ~12.3 hours
        - **Dataset**: Shakespeare's Complete Works

        ### Architecture Highlights
        - **Hidden Size**: 576
        - **Layers**: 30 (Deep & Narrow)
        - **Attention Heads**: 9
        - **KV Heads**: 3 (Grouped Query Attention)
        - **Context Length**: 2,048 tokens
        - **Vocabulary**: 50,257 tokens (GPT-2 tokenizer)

        ### Training Performance
        - Loss Reduction: 99.5% (11.28 → 0.053)
        - Convergence: Excellent (near-perfect memorization)
        - Notable: Loss spike at step 2,500 but recovered quickly
        """)

    # Main Interface
    with gr.Row():
        # Left column - Inputs
        with gr.Column(scale=1):
            prompt_input = gr.Textbox(
                label="Enter your prompt",
                placeholder="To be or not to be...",
                lines=3,
                value="To be or not to be"
            )

            with gr.Accordion("⚙️ Generation Settings", open=True):
                max_tokens_slider = gr.Slider(
                    minimum=10,
                    maximum=300,
                    value=100,
                    step=10,
                    label="Max Tokens",
                    info="Number of tokens to generate"
                )

                temperature_slider = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=0.8,
                    step=0.1,
                    label="Temperature",
                    info="Higher = more random, Lower = more deterministic"
                )

                top_k_slider = gr.Slider(
                    minimum=1,
                    maximum=100,
                    value=50,
                    step=1,
                    label="Top-K",
                    info="Smaller = more focused, Larger = more diverse"
                )

            generate_btn = gr.Button("🎭 Generate Shakespeare", variant="primary", size="lg")

            gr.Markdown("### 📝 Try these examples:")
            gr.Examples(
                examples=examples,
                inputs=[prompt_input, max_tokens_slider, temperature_slider, top_k_slider],
                label="Example Prompts"
            )

        # Right column - Output
        with gr.Column(scale=1):
            output_text = gr.Textbox(
                label="Generated Text",
                lines=15,
                placeholder="Your generated Shakespeare-style text will appear here...",
                show_copy_button=True
            )

            gr.Markdown("""
            ### 💡 Tips for Better Results
            - **Temperature 0.7-0.8**: More coherent, Shakespeare-like
            - **Temperature 0.9-1.2**: More creative, experimental
            - **Top-K 40-50**: Balanced diversity
            - **Max Tokens 80-150**: Good length for poetry/dialogue
            """)

    # Connect button to function
    generate_btn.click(
        fn=generate_shakespeare,
        inputs=[prompt_input, max_tokens_slider, temperature_slider, top_k_slider],
        outputs=output_text
    )

    # Footer
    gr.Markdown("""
    ---
    ### 🎓 About This Model
    This model was trained from scratch using PyTorch on Shakespeare's complete works.
    It demonstrates the SmolLM2-135M architecture with Grouped Query Attention (GQA),
    RoPE embeddings, and SwiGLU activations.

    **Training Achievement**: 99.5% loss reduction in just 5,500 steps!

    Made with ❤️ using PyTorch | [View on GitHub](#)
    """)

# ============================================================================
# LAUNCH
# ============================================================================

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
