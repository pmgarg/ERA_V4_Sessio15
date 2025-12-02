# 🚀 DeepSeek-V3 (174M) Implementation

A custom implementation of the **DeepSeek-V3** architecture, scaled down to ~174M parameters. This project demonstrates the power of **Multi-Head Latent Attention (MLA)** and **Mixture of Experts (MoE)** in a compact, efficient language model.

---

## 🧠 Architecture Overview

This model replaces the standard LLaMA-style architecture of **SmolLM2** with the advanced components of DeepSeek-V3.

### 🆚 DeepSeek-V3 vs. SmolLM2

| Feature | SmolLM2 (Baseline) | DeepSeek-V3 (Ours) | Impact |
| :--- | :--- | :--- | :--- |
| **Architecture** | Dense Transformer (LLaMA) | **MoE Transformer** | Sparse activation for efficiency |
| **Attention** | Grouped Query Attention (GQA) | **Multi-Head Latent Attention (MLA)** | KV compression (512 rank) reduces memory |
| **Feed-Forward** | Standard MLP (SwiGLU) | **DeepSeekMoE** | 1 Shared + 8 Routed Experts |
| **Active Params** | ~135M (100% active) | **~35M active** (per token) | Faster inference, higher capacity (~174M total) |
| **Positional Emb** | Standard RoPE | **Decoupled RoPE** | Better long-context handling |
| **Training Stability**| Standard Cross-Entropy | **Auxiliary Loss + Router Z-Loss** | Balanced expert load & stable routing |

---

## 🔑 Key Innovations

### 1. Multi-Head Latent Attention (MLA)
Instead of storing huge KV caches, MLA compresses Key-Value pairs into a low-rank latent vector (`kv_lora_rank=512`).
- **Benefit**: Drastically reduces inference memory usage while maintaining performance.
- **Mechanism**: Projects inputs down to a compressed latent space, then projects up for attention scores.

### 2. DeepSeekMoE (Mixture of Experts)
Uses a sophisticated routing strategy:
- **Shared Experts**: 1 expert is *always* active to capture common knowledge.
- **Routed Experts**: 8 experts available, with **Top-2** selected per token.
- **Benefit**: Allows the model to have a huge "brain" (174M params) but only use a small part of it for each word, making it efficient.

### 3. Training Stability (Z-Loss)
MoE models are notoriously hard to train. We implemented:
- **Auxiliary Loss**: Punishes the router if it over-uses one expert.
- **Router Z-Loss**: Stabilizes the logits entering the softmax, preventing numerical instability and "expert collapse".

---

## 📉 Training Progression & Loss

We trained the model in two phases on the `input-1.txt` dataset.

### Phase 1: The Learning Curve
*Initial instability was fixed by lowering LR and adding Z-Loss.*

| Step | Loss | Notes |
| :--- | :--- | :--- |
| **0** | `11.15` | Random initialization |
| **500** | `2.68` | Rapid initial convergence |
| **1000** | `0.20` | Strong pattern learning |
| **2000** | `0.10` | Fine-grained optimization |
| **3000** | `0.08` | Stable convergence |
| **4000** | `0.07` | Expert specialization |
| **5000** | **`0.065`** | **Phase 1 Complete** |

### Phase 2: Refinement
*Resumed from checkpoint for 500 extra steps.*

| Step | Loss | Status |
| :--- | :--- | :--- |
| **5001** | `0.064` | Resumed training |
| **5250** | `0.058` | Stable fine-tuning |
| **5500** | **`0.061`** | **Final Converged Model** |

> **Note**: The final loss of `~0.06` is achieved with a sparse architecture that is computationally cheaper during inference!

---

## 🛠️ How to Run

### 1. Web Interface (Flask)
Interact with the model using a beautiful web UI.

```bash
python app.py
```
*Open [http://localhost:7860](http://localhost:7860) in your browser.*

### 2. Training
Train the model from scratch using the notebook.

1. Open `test.ipynb`.
2. Run the **Phase 1** cells (5000 steps).
3. Run the **Phase 2** cells (500 steps).

---

## 📦 Project Structure

- `model.py`: Complete DeepSeek-V3 architecture (MLA, MoE, RoPE).
- `app.py`: Flask application for inference.
- `test.ipynb`: Training pipeline and verification.
- `requirements.txt`: Project dependencies.
- `checkpoint_5500.pt`: Trained model weights.

---

