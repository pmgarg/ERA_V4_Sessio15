import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Config
# -----------------------------

@dataclass
class DeepSeekV3Config:
    """
    Config for DeepSeek-V3 scaled down to ~135M parameters.
    """
    vocab_size: int = 50257
    hidden_size: int = 576
    num_hidden_layers: int = 30
    num_attention_heads: int = 9
    
    # MLA params
    kv_lora_rank: int = 512
    q_lora_rank: int = 0  # 0 means no compression for Query
    qk_rope_head_dim: int = 64
    v_head_dim: int = 64
    qk_nope_head_dim: int = 64
    
    # MoE params
    moe_intermediate_size: int = 160
    n_shared_experts: int = 1
    n_routed_experts: int = 8
    num_experts_per_tok: int = 2
    moe_layer_freq: int = 1  # Apply MoE every layer
    aux_loss_alpha: float = 0.001
    router_z_loss_coef: float = 0.001  # Coefficient for router z-loss
    seq_aux: bool = True
    
    # General
    max_position_embeddings: int = 2048
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    attention_dropout: float = 0.0
    initializer_range: float = 0.02
    tie_word_embeddings: bool = True
    torch_dtype: str = "bfloat16"
    use_cache: bool = True
    
    _extra: Dict[str, Any] = field(default_factory=dict, repr=False)

    def __init__(self, **kwargs):
        fields = {f.name: f for f in self.__dataclass_fields__.values()}
        extra = {}
        for k, v in kwargs.items():
            if k in fields:
                setattr(self, k, v)
            else:
                extra[k] = v
        for name, f in fields.items():
            if not hasattr(self, name):
                setattr(self, name, f.default)
        object.__setattr__(self, "_extra", extra)


# -----------------------------
# Utils
# -----------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(norm_x + self.eps)
        return self.weight * x


def get_activation(name: str):
    if name == "silu":
        return nn.SiLU()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_position_embeddings = max_position_embeddings
        self.dim = dim
        self._set_cos_sin_cache(max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len, device="cpu", dtype=torch.float32):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len, device=x.device, dtype=x.dtype)
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    # q, k: [bs, heads, seq_len, head_dim]
    # cos, sin: [seq_len, head_dim]
    
    if position_ids is None:
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
    else:
        # position_ids: [bs, seq_len]
        cos = cos[position_ids].unsqueeze(unsqueeze_dim)  # [bs, 1, seq_len, head_dim]
        sin = sin[position_ids].unsqueeze(unsqueeze_dim)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# -----------------------------
# DeepSeek-V3 Modules
# -----------------------------

class DeepSeekMLA(nn.Module):
    """
    Multi-Head Latent Attention (MLA)
    """
    def __init__(self, config: DeepSeekV3Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.kv_lora_rank = config.kv_lora_rank
        self.q_lora_rank = config.q_lora_rank
        
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.v_head_dim = config.v_head_dim
        
        self.q_head_dim = self.qk_rope_head_dim + self.qk_nope_head_dim
        
        # Query Projections
        if self.q_lora_rank > 0:
            self.q_down_proj = nn.Linear(self.hidden_size, self.q_lora_rank, bias=False)
            self.q_up_proj = nn.Linear(self.q_lora_rank, self.num_heads * self.q_head_dim, bias=False)
            self.q_norm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        else:
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.q_head_dim, bias=False)
            
        # KV Projections (MLA)
        self.kv_down_proj = nn.Linear(self.hidden_size, self.kv_lora_rank, bias=False)
        self.kv_norm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        
        # Up projection for Key (split into nope and rope parts)
        # K_nope: [kv_lora_rank] -> [num_heads * qk_nope_head_dim]
        # K_rope: [kv_lora_rank] -> [qk_rope_head_dim] (shared across heads for RoPE part usually, or per head)
        # DeepSeek V3 typically projects to num_heads * (nope + rope)
        # But for efficiency, K_rope is often shared or smaller. 
        # Here we implement standard MLA:
        # W_UK: projects compressed KV to K_nope (per head) + K_rope (per head)
        self.kv_up_proj = nn.Linear(self.kv_lora_rank, self.num_heads * (self.qk_nope_head_dim + self.qk_rope_head_dim), bias=False)
        
        # Value projection
        # W_UV: projects compressed KV to V (per head)
        self.kv_v_proj = nn.Linear(self.kv_lora_rank, self.num_heads * self.v_head_dim, bias=False)
        
        # Output projection
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, self.hidden_size, bias=False)
        
        self.rotary_emb = RotaryEmbedding(
            dim=self.qk_rope_head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ):
        bsz, q_len, _ = hidden_states.size()
        
        # 1. Query Generation
        if self.q_lora_rank > 0:
            q = self.q_down_proj(hidden_states)
            q = self.q_norm(q)
            q = self.q_up_proj(q)
        else:
            q = self.q_proj(hidden_states)
            
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim)
        q_nope, q_rope = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        
        # 2. KV Generation (Compressed)
        # In MLA, we compress inputs to a latent vector c_KV
        c_kv = self.kv_down_proj(hidden_states)
        c_kv = self.kv_norm(c_kv)
        
        # Generate K and V from compressed latent
        # k: [bs, seq, heads, nope+rope]
        k = self.kv_up_proj(c_kv)
        k = k.view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.qk_rope_head_dim)
        k_nope, k_rope = torch.split(k, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        
        # v: [bs, seq, heads, v_dim]
        v = self.kv_v_proj(c_kv)
        v = v.view(bsz, q_len, self.num_heads, self.v_head_dim)
        
        # 3. Apply RoPE
        # We only apply RoPE to the rope parts of Q and K
        kv_seq_len = v.shape[1]
        if kv_cache is not None:
            kv_seq_len += kv_cache[0].shape[1]
            
        cos, sin = self.rotary_emb(v, seq_len=kv_seq_len)
        
        # Transpose for RoPE: [bs, heads, seq, dim]
        q_rope = q_rope.transpose(1, 2)
        k_rope = k_rope.transpose(1, 2)
        q_nope = q_nope.transpose(1, 2)
        k_nope = k_nope.transpose(1, 2)
        v = v.transpose(1, 2)
        
        q_rope, k_rope = apply_rotary_pos_emb(q_rope, k_rope, cos, sin, position_ids)
        
        # Concatenate nope and rope parts back for attention
        # q: [bs, heads, seq, dim]
        q = torch.cat([q_nope, q_rope], dim=-1)
        k = torch.cat([k_nope, k_rope], dim=-1)
        
        # 4. KV Cache
        # For MLA, we ideally cache the compressed latent c_KV to save memory.
        # However, to keep this implementation compatible with standard attention loops 
        # and simpler, we will cache the projected K and V. 
        # (Optimized MLA caches c_KV and projects on the fly, but that requires custom kernels for speed).
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)
        
        new_kv_cache = (k, v) if use_cache else None
        
        # 5. Attention
        # [bs, heads, q_len, kv_len]
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.q_head_dim)
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
            
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)
        
        # [bs, q_len, heads, v_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.num_heads * self.v_head_dim)
        
        attn_output = self.o_proj(attn_output)
        
        return attn_output, new_kv_cache


class DeepSeekMoE(nn.Module):
    """
    DeepSeekMoE with Shared Experts and Routed Experts (Top-K)
    """
    def __init__(self, config: DeepSeekV3Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size
        self.n_shared_experts = config.n_shared_experts
        self.n_routed_experts = config.n_routed_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.aux_loss_alpha = config.aux_loss_alpha
        self.router_z_loss_coef = config.router_z_loss_coef
        self.seq_aux = config.seq_aux
        
        # Shared Experts
        self.shared_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_size, self.intermediate_size, bias=False),
                get_activation(config.hidden_act),
                nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
            )
            for _ in range(self.n_shared_experts)
        ])
        
        # Routed Experts
        self.routed_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_size, self.intermediate_size, bias=False),
                get_activation(config.hidden_act),
                nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
            )
            for _ in range(self.n_routed_experts)
        ])
        
        # Router
        self.router = nn.Linear(self.hidden_size, self.n_routed_experts, bias=False)

    def forward(self, x):
        # x: [bs, seq, hidden]
        bs, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)
        
        # 1. Shared Experts
        shared_output = 0
        for expert in self.shared_experts:
            shared_output = shared_output + expert(x_flat)
            
        # 2. Routed Experts
        router_logits = self.router(x_flat)  # [tokens, n_routed]
        routing_probs = F.softmax(router_logits, dim=-1)
        
        # Top-K selection
        top_k_probs, top_k_indices = torch.topk(routing_probs, self.num_experts_per_tok, dim=-1)
        # Normalize probs
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        routed_output = torch.zeros_like(x_flat)
        
        # Naive loop implementation for clarity (can be optimized with scatter/gather)
        # For small scale, this is fine.
        flat_indices = top_k_indices.view(-1)  # [tokens * k]
        flat_probs = top_k_probs.view(-1)      # [tokens * k]
        
        # We process each expert
        for i, expert in enumerate(self.routed_experts):
            # Find which tokens routed to this expert
            # mask: [tokens, k]
            mask = (top_k_indices == i)
            if mask.any():
                # Get inputs for this expert
                # We need to pick x_flat where mask is true. 
                # Since a token can go to multiple experts, we treat each assignment independently.
                
                # Indices of tokens that selected this expert
                batch_indices = torch.nonzero(mask, as_tuple=True)[0]
                
                # Get the corresponding probabilities
                # mask is boolean, so we can select from top_k_probs
                prob = top_k_probs[mask]
                
                expert_input = x_flat[batch_indices]
                expert_out = expert(expert_input)
                
                # Add to output (weighted)
                routed_output.index_add_(0, batch_indices, expert_out * prob.unsqueeze(-1))
                
        final_output = shared_output + routed_output
        final_output = final_output.view(bs, seq_len, hidden_dim)
        
        # 3. Aux Loss (Load Balancing) & Z-Loss
        # Switch between sequence-level or batch-level aux loss
        if self.training:
            # simple load balancing loss
            # target: uniform distribution
            # target_prob = 1.0 / self.n_routed_experts
            # actual usage
            expert_usage = routing_probs.mean(dim=0)  # [n_routed]
            aux_loss = self.aux_loss_alpha * torch.sum(expert_usage * expert_usage) * self.n_routed_experts
            
            # Router Z-Loss (stabilizes logits)
            # log(sum(exp(logits)))^2
            z_loss = torch.logsumexp(router_logits, dim=-1).pow(2).mean() * self.router_z_loss_coef
            
            aux_loss = aux_loss + z_loss
        else:
            aux_loss = 0.0
            
        return final_output, aux_loss


class DeepSeekV3DecoderLayer(nn.Module):
    def __init__(self, config: DeepSeekV3Config):
        super().__init__()
        self.self_attn = DeepSeekMLA(config)
        self.moe = DeepSeekMoE(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ):
        # Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_output, new_kv_cache = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
            use_cache=use_cache,
        )
        hidden_states = residual + attn_output
        
        # MoE
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        moe_output, aux_loss = self.moe(hidden_states)
        hidden_states = residual + moe_output
        
        return hidden_states, new_kv_cache, aux_loss


# -----------------------------
# Model Wrapper
# -----------------------------

class DeepSeekV3Model(nn.Module):
    def __init__(self, config: DeepSeekV3Config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [DeepSeekV3DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        self._reset_parameters()

    def _reset_parameters(self):
        for name, param in self.named_parameters():
            if param.dim() == 1:
                nn.init.ones_(param) if "weight" in name else nn.init.zeros_(param)
            else:
                if "router" in name:
                    nn.init.normal_(param, mean=0.0, std=0.01) # Smaller init for router
                else:
                    nn.init.normal_(param, mean=0.0, std=self.config.initializer_range)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        use_cache: bool = False,
    ):
        device = input_ids.device
        bsz, seq_len = input_ids.shape
        
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
            
        # Prepare mask
        if attention_mask is None:
            # Causal mask only
            mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
            mask = torch.triu(mask, diagonal=1)
            attention_mask = mask.unsqueeze(0).unsqueeze(0) # [1, 1, seq, seq]
        else:
            # If mask provided (e.g. padding), expand it
            # Assuming attention_mask is [bs, seq] with 1 for keep, 0 for pad
            # We need to combine with causal mask
            causal_mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
            causal_mask = torch.triu(causal_mask, diagonal=1)
            
            # Expand padding mask: 1 -> 0, 0 -> -inf
            pad_mask = (1.0 - attention_mask) * -1e9
            pad_mask = pad_mask.unsqueeze(1).unsqueeze(1) # [bs, 1, 1, seq]
            
            attention_mask = causal_mask.unsqueeze(0).unsqueeze(0) + pad_mask

        hidden_states = self.embed_tokens(input_ids)
        
        new_kv_cache = [] if use_cache else None
        if kv_cache is None:
            kv_cache = (None,) * len(self.layers)
            
        total_aux_loss = 0.0
        
        for layer, layer_kv in zip(self.layers, kv_cache):
            hidden_states, layer_cache, layer_aux_loss = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache=layer_kv,
                use_cache=use_cache
            )
            total_aux_loss += layer_aux_loss
            if use_cache:
                new_kv_cache.append(layer_cache)
                
        hidden_states = self.norm(hidden_states)
        
        if use_cache:
            new_kv_cache = tuple(new_kv_cache)
            
        return hidden_states, new_kv_cache, total_aux_loss


class DeepSeekV3ForCausalLM(nn.Module):
    def __init__(self, config: DeepSeekV3Config):
        super().__init__()
        self.config = config
        self.model = DeepSeekV3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        use_cache: bool = False,
    ):
        hidden_states, new_kv_cache, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
            use_cache=use_cache,
        )
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            # Add aux loss
            loss = loss + aux_loss

        return {
            "loss": loss,
            "logits": logits,
            "kv_cache": new_kv_cache,
            "aux_loss": aux_loss
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 32,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            out = self(
                input_ids=input_ids,
                use_cache=False
            )
            logits = out["logits"][:, -1, :]
            
            if temperature != 1.0:
                logits = logits / temperature
                
            if top_k is not None and top_k > 0:
                values, _ = torch.topk(logits, top_k)
                min_values = values[:, -1].unsqueeze(-1)
                logits = torch.where(logits < min_values, torch.full_like(logits, float("-inf")), logits)
                
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
        return input_ids
