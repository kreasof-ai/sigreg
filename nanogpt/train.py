import os
import math
import time
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cut_cross_entropy import linear_cross_entropy
from huggingface_hub import PyTorchModelHubMixin  # NEW: For HF Hub integration
import json

# ==========================================
# 1. Configuration
# ==========================================
# OPTIONS: 'baseline', 'weak', 'strong', 'discrete', 'zipfian'
REG_MODE = 'zipfian'
SIGR_ALPHA = 0.01     # Physics constraint strength
SKETCH_DIM = 64        # Dimension of the random observer

# Optimization Hyperparams
BATCH_SIZE = 16        # Lower than vision because SeqLen is long
SEQ_LEN = 1024         # Context length
LEARNING_RATE = 0.02   # Muon likes high LR
EPOCHS = 1             # 1 Epoch on FineWeb is plenty for testing
GRAD_ACCUM = 4         # Simulate larger batch size
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data Config
DATA_DIR = "finewebedu10B" # Path where cached_fineweb10B.py saved .bin files
VOCAB_SIZE = 50304           # GPT-2 vocab size (rounded to 128 multiple)

# HuggingFace Hub Config (NEW)
HF_REPO_ID = "username/repo"  # Change to your HF username/repo
SAVE_DIR = "./model_checkpoints"             # Local save directory
UPLOAD_TO_HF = True                          # Set False to skip upload

import wandb
wandb.login() # You'll need an API key

wandb.init(
    project="nanogpt",
    name=f"{REG_MODE}_1B",
    config={
        "batch_size": BATCH_SIZE,
        "reg_mode": REG_MODE,
        "sigreg_alpha": SIGR_ALPHA,
        "sketch_dim": SKETCH_DIM,
        "optimizer": "Muon",
        "hf_repo_id": HF_REPO_ID,
    }
)

print(f"GPT Training on: {DEVICE} | Mode: {REG_MODE} | Alpha: {SIGR_ALPHA}")

# ==========================================
# 2. Physics Engine (SIGReg)
# ==========================================

def sigreg_weak_loss(x, sketch_dim=64):
    """
    Forces Covariance(x) ~ Identity via Sketching.
    """

    N, C = x.size()

    # 1. Sketching
    if C > sketch_dim:
        # Generate random matrix S
        S = torch.randn(sketch_dim, C, device=x.device) / (C ** 0.5)
        x = x @ S.T  # [N, sketch_dim]
    else:
        sketch_dim = C

    # 2. Centering
    x = x - x.mean(dim=0, keepdim=True)

    # 3. Covariance
    cov = (x.T @ x) / (N - 1 + 1e-6)

    # 4. Target Identity
    target = torch.eye(sketch_dim, device=x.device)

    # 5. Frobenius Norm
    loss = torch.norm(cov - target, p='fro')

    return loss
    
def zipf_orthogonal_est(x, sketch_dim=64, zipf_s=1.0, lam_ang=1.0, lam_mag=1.0, eps=1e-6):
    """
    Keeps directions approximately orthogonal / decorrelated,
    while forcing the sample magnitudes to follow a Zipf-like decay.

    zipf_s: Zipf exponent (higher = steeper decay)
    """

    N, C = x.size()

    # 1) Optional sketching
    if C > sketch_dim:
        S = torch.randn(sketch_dim, C, device=x.device) / (C ** 0.5)
        x = x @ S.T
        C = sketch_dim

    # 2) Center
    x = x - x.mean(dim=0, keepdim=True)

    # 3) Split into direction + magnitude
    norms = x.norm(dim=1, keepdim=True).clamp_min(eps)
    u = x / norms   # direction-only vectors

    # 4) Angular orthogonality / decorrelation loss
    #    Suppress off-diagonal cosine correlations
    G = (u.T @ u) / (N - 1 + eps)
    ang_loss = torch.norm(G - torch.diag(torch.diag(G)), p='fro')

    # 5) Zipf magnitude loss
    #    Zipf is rank-based, so compare sorted norms to rank^(-s)
    sorted_norms, _ = torch.sort(norms.squeeze(-1), descending=True)
    ranks = torch.arange(1, N + 1, device=x.device, dtype=x.dtype)

    zipf_target = ranks.pow(-zipf_s)
    zipf_target = zipf_target / (zipf_target.sum() + eps)

    sorted_norms = sorted_norms / (sorted_norms.sum() + eps)

    mag_loss = torch.norm(sorted_norms - zipf_target, p='fro')

    return lam_ang * ang_loss + lam_mag * mag_loss

def sireg_discrete_loss(x, sketch_dim=64):
    """
    Discrete-SIReg: Forces directional orthogonality (Static Entropy) 
    while allowing magnitudes (outlier tokens) to grow freely.
    """

    N, C = x.size()

    # -----------------------------------------------------------
    # Normalize to the unit sphere, then stretch the 
    # sphere so its radius is sqrt(C). 
    # This brings the variance per-dimension exactly back to 1.0!
    # -----------------------------------------------------------
    x = F.normalize(x, p=2, dim=-1) * (C ** 0.5)

    # 1. Sketching
    if C > sketch_dim:
        # Generate random matrix S
        S = torch.randn(sketch_dim, C, device=x.device) / (C ** 0.5)
        x = x @ S.T  # [N, sketch_dim]
    else:
        sketch_dim = C

    # 2. Centering
    x = x - x.mean(dim=0, keepdim=True)

    # 3. Covariance
    cov = (x.T @ x) / (N - 1 + 1e-6)

    # 4. Target Identity
    target = torch.eye(sketch_dim, device=x.device)

    # 5. Frobenius Norm
    loss = torch.norm(cov - target, p='fro')

    return loss

def sigreg_strong_loss(x, sketch_dim=64):
    """
    Forces ECF(x) ~ ECF(Gaussian). Matches ALL moments.
    """
    
    N, C = x.size()

    # 1. Projection
    A = torch.randn(C, sketch_dim, device=x.device)
    A = A / (A.norm(p=2, dim=0, keepdim=True) + 1e-6)

    # 2. Integration Points
    t = torch.linspace(-5, 5, 17, device=x.device)
    exp_f = torch.exp(-0.5 * t**2)

    # 3. Empirical CF (Rewritten to avoid complex numbers for torch.compile support)
    proj = x @ A
    args = proj.unsqueeze(2) * t.view(1, 1, -1)

    # exp(ix) = cos(x) + i*sin(x)
    # E[exp(ix)] = E[cos(x)] + i*E[sin(x)]
    cos_mean = torch.cos(args).mean(dim=0)
    sin_mean = torch.sin(args).mean(dim=0)

    # 4. Loss
    # |ECF - Target|^2 = |(Real - Target) + i(Imag)|^2 = (Real - Target)^2 + Imag^2
    # Target (exp_f) is real-valued.
    diff_sq = (cos_mean - exp_f.unsqueeze(0)).square() + sin_mean.square()

    err = diff_sq * exp_f.unsqueeze(0)

    loss = torch.trapz(err, t, dim=1) * N
    return loss.mean()

# ==========================================
# 3. Modern Transformer Architecture
# ==========================================

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len, dtype=inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    def forward(self, x, seq_len):
        return self.cos_cached[:, :, :seq_len, :], self.sin_cached[:, :, :seq_len, :]

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads

        # Calculate repetition factor for GQA
        self.num_q_per_kv = num_heads // num_kv_heads

        # Ensure num_heads is divisible by num_kv_heads
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"

        # QK-Norm (Critical for Muon stability)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

        # Separate projections
        self.q_proj = nn.Linear(dim, dim, bias=False)  # Query projection
        self.kv_proj = nn.Linear(dim, 2 * num_kv_heads * self.head_dim, bias=False)  # Combined KV projection
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x, rope):
        B, T, C = x.size()

        # Project queries
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim)

        # Project keys and values together
        kv = self.kv_proj(x).view(B, T, 2, self.num_kv_heads, self.head_dim)
        k, v = kv[:, :, 0], kv[:, :, 1]  # (B, T, num_kv_heads, head_dim)

        # Reshape for attention: (B, num_heads, T, head_dim)
        q = q.transpose(1, 2)  # (B, num_heads, T, head_dim)
        k = k.transpose(1, 2)  # (B, num_kv_heads, T, head_dim)
        v = v.transpose(1, 2)  # (B, num_kv_heads, T, head_dim)

        # Apply QK-Norm
        q, k = self.q_norm(q), self.k_norm(k)

        # Apply RoPE
        cos, sin = rope(v, T)
        q, k = apply_rope(q, k, cos, sin)

        # Repeat K and V for GQA (if num_kv_heads != num_heads)
        if self.num_kv_heads != self.num_heads:
            k = k.repeat_interleave(self.num_q_per_kv, dim=1)
            v = v.repeat_interleave(self.num_q_per_kv, dim=1)

        # Flash Attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C)

        return self.proj(y)

class MLP(nn.Module):
    def __init__(self, dim, expand=3):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * expand, bias=False)
        self.fc2 = nn.Linear(dim * expand, dim, bias=False)

    def forward(self, x):
        return self.fc2(F.silu(self.fc1(x)))

class Block(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads)
        self.norm2 = RMSNorm(dim)
        self.mlp = MLP(dim)

    def forward(self, x, rope):
        # Attention Residual
        attn = self.attn(self.norm1(x), rope)
        x = x + attn

        # MLP Residual
        # SIGReg Injection Point: Regulate the residual stream
        # This treats tokens as particles drifting in semantic space
        mlp_out = self.mlp(self.norm2(x))
        x = x + mlp_out

        reg_loss = torch.tensor(0.0, device=x.device)
        if REG_MODE != 'baseline':
            batch_size, seq_len, hidden_dim = x.shape
            flat_rep = x.reshape(-1, hidden_dim) # new shape: [batch_size * seq_len, hidden_dim]

            if REG_MODE == 'weak':
                reg_loss = sigreg_weak_loss(flat_rep, SKETCH_DIM)
            elif REG_MODE == 'strong':
                reg_loss = sigreg_strong_loss(flat_rep, SKETCH_DIM)
            elif REG_MODE == 'discrete':
                reg_loss = sireg_discrete_loss(flat_rep, SKETCH_DIM)
            elif REG_MODE == 'zipfian':
                reg_loss = zipf_orthogonal_est(flat_rep, SKETCH_DIM)

        return x, reg_loss

# ==========================================
# MODERN GPT WITH HUGGINGFACE HUB MIXIN (NEW)
# ==========================================
class ModernGPT(
    nn.Module, 
    PyTorchModelHubMixin,  # NEW: Enables push_to_hub() and from_pretrained()
    library_name="nanogpt-sigreg",
):
    def __init__(self, vocab_size, dim=768, depth=12, heads=12, kv_heads=12):
        super().__init__()
        self.vocab_size = vocab_size  # NEW: Store for config
        self.dim = dim                # NEW: Store for config
        self.depth = depth            # NEW: Store for config
        self.heads = heads            # NEW: Store for config
        self.kv_heads = kv_heads      # NEW: Store for config
        
        self.embed = nn.Embedding(vocab_size, dim)
        self.rope = RotaryEmbedding(dim // heads)
        self.blocks = nn.ModuleList([Block(dim, heads, kv_heads) for _ in range(depth)])
        self.norm = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

        # Weight Tying
        self.embed.weight = self.head.weight

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, x, targets=None):
        x = self.embed(x)

        total_phys_loss = 0.0
        for block in self.blocks:
            x, p_loss = block(x, self.rope)
            total_phys_loss += p_loss

        x = self.norm(x)

        logits = None
        loss = None
        if targets is not None:
            # classifier shape expected: (vocab_size, dim)
            classifier = self.head.weight

            # compute token-level loss that predicts next-token (causal)
            loss = linear_cross_entropy(x, classifier, targets)
        else:
            logits = self.head(x)

        return logits, loss, (total_phys_loss / len(self.blocks))
    
    # NEW: Generate config for HF Hub
    def _generate_config(self):
        return {
            "vocab_size": self.vocab_size,
            "dim": self.dim,
            "depth": self.depth,
            "heads": self.heads,
            "kv_heads": self.kv_heads,
            "reg_mode": REG_MODE,
            "sigreg_alpha": SIGR_ALPHA,
            "sketch_dim": SKETCH_DIM,
        }

# ==========================================
# 4. Muon Optimizer (Standalone Compact Version)
# ==========================================
def zeropower_via_newtonschulz5(G, steps=5):
    """
    Newton-Schulz iteration to compute the zeroth power of a matrix (orthogonalization).
    """
    X = G.bfloat16()
    X = X / (X.norm() + 1e-7) # Approx spectral norm
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = A @ A
        # Quintic iteration coefficients
        X = 3.4445 * X - 4.7750 * (A @ X) + 2.0315 * (B @ X)
    if G.size(0) > G.size(1):
        X = X.T
    return X.type_as(G)

class Muon(torch.optim.Optimizer):
    """
    Muon - Momentum Orthogonalized by Newton-Schulz.
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            ns_steps = group['ns_steps']
            for p in group['params']: # Muon only works on 2D matrices
                if p.grad is None: continue
                g = p.grad
                if g.ndim != 2: continue

                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p)

                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)

                if group['nesterov']:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf

                # Orthogonalize update
                g_orth = zeropower_via_newtonschulz5(g, steps=ns_steps)

                # Update weights
                p.data.add_(g_orth, alpha=-lr)

# ==========================================
# 5. Data Loader (Reading .bin files)
# ==========================================
class SimpleBinLoader:
    def __init__(self, data_dir, batch_size, seq_len, split='train'):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.split = split

        # Pattern match based on NanoGPT file naming convention
        pattern = os.path.join(data_dir, f"finewebedu_{split}_*.bin")
        self.files = sorted(glob.glob(pattern))

        if not self.files:
            print(f"WARNING: No {split} files found in {data_dir}. Looking for generic .bin...")
            # Fallback for generic testing if specific names aren't found
            all_files = sorted(glob.glob(os.path.join(data_dir, "*.bin")))
            if split == 'train':
                self.files = all_files[:-1] # Use all but last for train
            else:
                self.files = all_files[-1:] # Use last for val

        assert len(self.files) > 0, f"No files found for split {split}"
        print(f"Found {len(self.files)} files for {split} split.")

        # Calculate total tokens to set accurate num_batches
        total_tokens = 0
        for fname in self.files:
            sz = os.path.getsize(fname)
            # Subtract header (1024 bytes), div by 2 (uint16)
            total_tokens += (sz - 1024) // 2

        self.num_batches = total_tokens // (batch_size * seq_len)
        print(f"Total tokens in {split}: {total_tokens} | Total batches: {self.num_batches}")

        self.current_shard = 0
        self.data = self._load_data(self.files[self.current_shard])
        self.ptr = 0

    def _load_data(self, filename):
        with open(filename, "rb") as f:
            # Skip header (256 * 4 bytes)
            f.seek(256 * 4)
            # Read tokens (uint16) and convert to int64 (long)
            tokens = np.frombuffer(f.read(), dtype=np.uint16)
        return torch.from_numpy(tokens.astype(np.int64))

    def next_batch(self):
        # We need (Seq_Len + 1) tokens per sequence to have targets for every input
        buf_size = self.batch_size * (self.seq_len + 1)

        # Check if we have enough data in the current shard
        if self.ptr + buf_size > len(self.data):
            # Advance shard
            self.current_shard = (self.current_shard + 1) % len(self.files)
            print(f"Loading shard {self.files[self.current_shard]}")
            self.data = self._load_data(self.files[self.current_shard])
            self.ptr = 0

        chunk = self.data[self.ptr : self.ptr + buf_size]

        # Advance pointer.
        self.ptr += buf_size

        # Reshape
        chunk = chunk.view(self.batch_size, self.seq_len + 1)

        # x is [B, T] (all columns except the last)
        x = chunk[:, :-1]
        # y is [B, T] (all columns except the first)
        y = chunk[:, 1:]

        return x.to(DEVICE), y.to(DEVICE)

@torch.no_grad()
def estimate_loss(model, loader, eval_iters):
    out = {}
    model.eval()
    losses_task = []
    losses_phys = []

    print(f"Evaluating on {loader.split} set...")
    for _ in range(eval_iters):
        X, Y = loader.next_batch()
        # Note: We don't scale the loss here, we just want to observe the raw values
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            _, loss_task, loss_phys = model(X, Y)
        losses_task.append(loss_task.item())
        losses_phys.append(loss_phys.item())

    out["task"] = sum(losses_task) / len(losses_task)
    out["phys"] = sum(losses_phys) / len(losses_phys)
    model.train()
    return out

def get_lr(step, max_lr, min_lr, warmup_steps, max_steps):
    # 1. Linear Warmup
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    # 2. Constant Min LR after max_steps
    if step > max_steps:
        return min_lr
    # 3. Cosine Decay
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

# ==========================================
# NEW: Save and Upload Functions
# ==========================================
def save_model_checkpoint(model, save_dir, step, val_loss):
    """Save model weights locally with metadata"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Create checkpoint filename with step and loss
    checkpoint_name = f"model_step{step}_loss{val_loss:.4f}"
    checkpoint_path = os.path.join(save_dir, checkpoint_name)
    
    # Save using PyTorchModelHubMixin's save_pretrained
    model.save_pretrained(checkpoint_path)
    
    # Save additional training metadata
    metadata = {
        "step": step,
        "val_loss": val_loss,
        "reg_mode": REG_MODE,
        "sigreg_alpha": SIGR_ALPHA,
        "sketch_dim": SKETCH_DIM,
        "batch_size": BATCH_SIZE,
        "seq_len": SEQ_LEN,
        "learning_rate": LEARNING_RATE,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    with open(os.path.join(checkpoint_path, "training_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Saved checkpoint to: {checkpoint_path}")
    return checkpoint_path

def upload_to_huggingface(model, repo_id, commit_message="Upload model weights"):
    """Upload model to HuggingFace Hub"""
    try:
        print(f"Uploading to HuggingFace Hub: {repo_id}...")
        
        # Generate and save config
        config = model._generate_config()
        
        # Push to hub using PyTorchModelHubMixin's push_to_hub
        model.push_to_hub(
            repo_id=repo_id,
            config=config,
            commit_message=commit_message,
            private=False,  # Set True for private repo
        )
        
        print(f"✓ Successfully uploaded to https://huggingface.co/{repo_id}")
        return True
        
    except Exception as e:
        print(f"✗ Upload failed: {e}")
        print("Make sure you're logged in with: huggingface-cli login")
        return False

# ==========================================
# 6. Main Loop
# ==========================================
if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')

    # --- 1. Initialize Loaders ---
    # Make sure you ran: python data/cached_fineweb10B.py 2 (to get at least 1 train and 1 val shard)
    train_loader = SimpleBinLoader(DATA_DIR, BATCH_SIZE, SEQ_LEN, split='train')
    val_loader   = SimpleBinLoader(DATA_DIR, BATCH_SIZE, SEQ_LEN, split='val')

    # --- 2. Initialize Model ---
    print("Initializing ModernGPT...")
    model = ModernGPT(vocab_size=VOCAB_SIZE, dim=640, depth=24, heads=10, kv_heads=2)
    model.to(DEVICE)

    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    model = torch.compile(model)

    # --- 3. Optimizers ---
    # Separate 2D params (Muon) from 1D/Scalars (AdamW)
    muon_params = [p for p in model.parameters() if p.ndim == 2]
    adam_params = [p for p in model.parameters() if p.ndim != 2]

    optim_muon = Muon(muon_params, lr=LEARNING_RATE, momentum=0.95)
    optim_adam = torch.optim.AdamW(adam_params, lr=LEARNING_RATE/10, betas=(0.9, 0.95), weight_decay=0.01)

    scaler = torch.cuda.amp.GradScaler()

    # --- 4. Training Loop ---
    print(f"Starting Training | Mode: {REG_MODE} | Alpha: {SIGR_ALPHA}")
    t0 = time.time()

    EVAL_INTERVAL = 500
    EVAL_ITERS = 50
    MAX_STEPS = train_loader.num_batches * EPOCHS
    WARMUP_STEPS = int(MAX_STEPS * 0.05) # 5% warmup

    # Base LRs
    MUON_LR_BASE = LEARNING_RATE
    ADAM_LR_BASE = LEARNING_RATE / 10

    start_time = time.time()
    
    # Track best model for upload
    best_val_loss = float('inf')
    best_step = 0

    for step in range(MAX_STEPS):

        # --- A. Update Learning Rate ---
        # Calculate the multiplier (0.0 to 1.0) based on schedule
        # We use a dummy max_lr=1.0 so we get a scaling factor back
        lr_factor = get_lr(step, max_lr=1.0, min_lr=0.1, warmup_steps=WARMUP_STEPS, max_steps=MAX_STEPS)

        # Apply to Muon
        for param_group in optim_muon.param_groups:
            param_group['lr'] = MUON_LR_BASE * lr_factor

        # Apply to AdamW
        for param_group in optim_adam.param_groups:
            param_group['lr'] = ADAM_LR_BASE * lr_factor

        # --- B. Validation ---
        if step % EVAL_INTERVAL == 0 and step > 0:
            dt = time.time() - t0
            metrics = estimate_loss(model, val_loader, EVAL_ITERS)
            if metrics['task'] < best_val_loss:          
                best_val_loss = metrics['task']           
                best_step = step  
            print(f"Step {step}: Val Loss: {metrics['task']:.4f} | Phys Loss: {metrics['phys']:.4f} | LR: {MUON_LR_BASE * lr_factor:.5f} | Time: {dt:.2f}s")
            t0 = time.time()

            wandb.log({
              "step": step,
              "val_loss": metrics['task'],
            })

        # --- C. Training Step ---
        optim_adam.zero_grad(set_to_none=True)
        optim_muon.zero_grad(set_to_none=True)

        for micro_step in range(GRAD_ACCUM):
            x, y = train_loader.next_batch()

            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                _, loss_task, loss_phys = model(x, y)

                loss = (1 - SIGR_ALPHA) * loss_task + (SIGR_ALPHA * loss_phys)
                loss = loss / GRAD_ACCUM

            scaler.scale(loss).backward()

        scaler.unscale_(optim_adam)
        inv_scale = 1.0 / scaler.get_scale()
        for p in muon_params:
            if p.grad is not None:
                p.grad.mul_(inv_scale)
                           
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optim_adam.step()
        optim_muon.step()
        scaler.update()

        if step % 100 == 0:
            print(f"Iter {step}: Task {loss_task.item():.4f} | Phys {loss_phys.item():.4f} | LR {MUON_LR_BASE * lr_factor:.4f}")

            wandb.log({
              "step": step,
              "train_loss": loss_task.item(),
              "phys_loss": loss_phys.item(),
              "learning_rate": MUON_LR_BASE * lr_factor
            })

    # ==========================================
    # FINAL: Save and Upload Model
    # ==========================================
    print("\n" + "="*60)
    print("TRAINING COMPLETE - SAVING AND UPLOADING MODEL")
    print("="*60)
    
    # Final evaluation
    metrics = estimate_loss(model, val_loader, EVAL_ITERS)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"FINAL: Val Loss: {metrics['task']:.4f} | Phys Loss: {metrics['phys']:.4f} | Time: {elapsed_time:.2f}s")
    
    # Save final model checkpoint
    final_checkpoint = save_model_checkpoint(
        model, 
        SAVE_DIR, 
        step=MAX_STEPS-1, 
        val_loss=metrics['task']
    )
    
    # Upload to HuggingFace Hub if enabled
    if UPLOAD_TO_HF:
        print("\n" + "-"*60)
        print("UPLOADING TO HUGGINGFACE HUB")
        print("-"*60)
        
        # Make sure you're logged in: huggingface-cli login
        upload_success = upload_to_huggingface(
            model, 
            HF_REPO_ID, 
            commit_message=f"NanoGPT-SIGReg {REG_MODE} - Val Loss: {metrics['task']:.4f}"
        )
        
        if upload_success:
            print(f"\n🎉 Model available at: https://huggingface.co/{HF_REPO_ID}")
            
            # Log to wandb
            wandb.log({
                "hf_repo_id": HF_REPO_ID,
                "hf_url": f"https://huggingface.co/{HF_REPO_ID}",
                "final_val_loss": metrics['task'],
                "best_val_loss": best_val_loss,
                "best_step": best_step,
            })
    
    print("\n" + "="*60)
    print("DONE!")
    print("="*60)