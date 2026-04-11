import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from huggingface_hub import hf_hub_download
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. Global Configurations
# ==========================================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if torch.backends.mps.is_available(): DEVICE = 'mps'

# ViT Architecture Configs (must match training exactly)
DROPOUT = 0.0
IMG_SIZE = 224
PATCH_SIZE = 16
NUM_CLASSES = 1000
SKETCH_DIM = 64

# ------------------------------------------
# Physics Engine: The Regularizers
# ------------------------------------------

def sigreg_weak_loss(x, sketch_dim=64):
    """
    Forces Covariance(x) ~ Identity.
    Matches the 2nd Moment (Spherical Cloud).
    """
    N, C = x.size()

    # 1. Sketching
    S = torch.randn(sketch_dim, C, device=x.device) / (C ** 0.5)
    x = x @ S.T  # [N, sketch_dim]

    # 2. Centering & Covariance
    x = x - x.mean(dim=0, keepdim=True)
    cov = (x.T @ x) / (N - 1 + 1e-6)

    # 3. Target Identity
    target = torch.eye(sketch_dim, device=x.device)

    # 4. Off-diagonal suppression + Diagonal maintenance
    return torch.norm(cov - target, p='fro')

def sigreg_strong_loss(x, sketch_dim=64):
    """
    Forces ECF(x) ~ ECF(Gaussian).
    Matches ALL Moments (Maximum Entropy Cloud).
    Exact implementation of LeJEPA Algorithm 1.
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
# 2. Physics Model Definitions
# ==========================================
import torch
import torch.nn as nn
import torch.nn.functional as F

class RotaryEmbedding2D(nn.Module):
    def __init__(self, dim, max_shape=(32, 32)):
        super().__init__()
        self.dim = dim
        # We split dim into two for x and y frequencies
        self.dim_x = dim // 2
        self.dim_y = dim - self.dim_x

        # Precompute frequencies
        inv_freq_x = 1.0 / (10000 ** (torch.arange(0, self.dim_x, 2).float() / self.dim_x))
        inv_freq_y = 1.0 / (10000 ** (torch.arange(0, self.dim_y, 2).float() / self.dim_y))

        self.register_buffer("inv_freq_x", inv_freq_x)
        self.register_buffer("inv_freq_y", inv_freq_y)

    def forward(self, h, w, device):
        # Generate grid
        seq_y = torch.arange(h, device=device, dtype=self.inv_freq_y.dtype)
        seq_x = torch.arange(w, device=device, dtype=self.inv_freq_x.dtype)

        # Outer product to get (H, W, dim/2)
        freqs_x = torch.einsum("i,j->ij", seq_x, self.inv_freq_x)
        freqs_y = torch.einsum("i,j->ij", seq_y, self.inv_freq_y)

        # Combine to (H, W, dim/2) -> repeat for cos/sin format
        emb_x = torch.cat((freqs_x, freqs_x), dim=-1)
        emb_y = torch.cat((freqs_y, freqs_y), dim=-1)

        # We need to construct the full 2D embeddings
        # Assuming we split the head dim: [x_part, y_part]
        # We broaden to fit the sequence length
        # Result shape: [H*W, 1, Dim] for broadcasting

        # Broadcast x along height, y along width
        # freqs_x: [W, dim_x] -> [H, W, dim_x]
        emb_x = emb_x.unsqueeze(0).repeat(h, 1, 1)
        # freqs_y: [H, dim_y] -> [H, W, dim_y]
        emb_y = emb_y.unsqueeze(1).repeat(1, w, 1)

        # Concatenate x and y frequencies: [H, W, dim]
        freqs = torch.cat([emb_x, emb_y], dim=-1)

        # Flatten: [H*W, dim]
        freqs = freqs.flatten(0, 1)
        return freqs[None, :, :] # [1, Seq, Dim]

def apply_rotary_pos_emb(q, k, freqs):
    # q, k: [B, H, Seq, Dim]
    # freqs: [1, Seq, Dim]

    # Split into pairs for rotation
    q_len = q.shape[-1]

    # Cos/Sin
    cos = freqs.cos()
    sin = freqs.sin()

    # Apply rotation
    # (x, y) -> (x cos - y sin, x sin + y cos)
    # Standard rotate_half implementation
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

import torch.nn.functional as F

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training: return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
    def forward(self, x): return drop_path(x, self.drop_prob, self.training)

class ThermoAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # Note: self.scale is handled automatically by SDPA, but good to keep if needed manually
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(DROPOUT)

        # RoPE generator
        self.rope = RotaryEmbedding2D(head_dim)

        self.q_norm = nn.RMSNorm(head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(head_dim, eps=1e-6)

    def forward(self, x, H, W):
        B, N, C = x.shape
        # Shape: [3, B, Heads, SeqLen, HeadDim]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = self.q_norm(q)
        k = self.k_norm(k)

        cls_q = q[:, :, :1]
        cls_k = k[:, :, :1]

        q = q[:, :, 1:]
        k = k[:, :, 1:]

        # --- Apply 2D RoPE ---
        # RoPE modifies Q and K in place or returns new tensors.
        # It operates on the HeadDim, so it's compatible with the split heads.
        freqs = self.rope(H, W, x.device) # [1, SeqLen, HeadDim]
        q, k = apply_rotary_pos_emb(q, k, freqs)
        # ---------------------

        q = torch.cat((cls_q, q), dim=2)
        k = torch.cat((cls_k, k), dim=2)

        # --- Flash Attention ---
        # PyTorch 2.0+ automatically optimizes this using FlashAttention v2 on CUDA.
        # Input shapes are already (Batch, Heads, SeqLen, Dim), which SDPA expects.
        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=DROPOUT,
            is_causal=False  # ViT is bidirectional, not causal like GPT
        )

        # Reshape back: [B, Heads, N, Dim] -> [B, N, C]
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        # ---------------------
        return x

class ThermoViTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., reg_mode='baseline', sketch_dim=64):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim, eps=1e-6)
        self.attn = ThermoAttention(dim, num_heads=num_heads)
        self.norm2 = nn.RMSNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(DROPOUT),
        )
        self.reg_mode = reg_mode
        self.sketch_dim = sketch_dim

        self.drop_path = DropPath(DROPOUT) if DROPOUT > 0. else nn.Identity()

        self.gamma_1 = nn.Parameter(1e-4 * torch.ones((dim)),requires_grad=True)
        self.gamma_2 = nn.Parameter(1e-4 * torch.ones((dim)),requires_grad=True)

    def forward(self, x, H, W):
        # Attention Residual
        x = x + self.gamma_1 * self.drop_path(self.attn(self.norm1(x), H, W))

        # MLP Residual
        # Note: We apply SIGReg AFTER the block computation but BEFORE the next block.
        # This keeps the "Residual Stream" clean and Gaussian.

        x = x + self.gamma_2 * self.drop_path(self.mlp(self.norm2(x)))

        # --- PHYSICS INJECTION ---
        reg_loss = torch.tensor(0.0, device=x.device)
        if self.reg_mode != 'baseline':
            # Global Average Pool of the tokens [B, N, C] -> [B, C]
            # This represents the "Image Vector" at this depth
            flat_rep = x.mean(dim=1)

            # Crucial: Pre-Norm vs Post-Norm context.
            # LayerNorm forces variance=1. SIGReg forces Distribution=Gaussian.
            # They are compatible.
            if self.reg_mode == 'weak':
                reg_loss = sigreg_weak_loss(flat_rep, self.sketch_dim)
            elif self.reg_mode == 'strong':
                reg_loss = sigreg_strong_loss(flat_rep, self.sketch_dim)

        return x, reg_loss

class ThermoViT(nn.Module):
    def __init__(self, img_size=32, patch_size=4, num_classes=100,
                 dim=384, depth=12, heads=12, mlp_ratio=4,
                 reg_mode='strong', sketch_dim=64):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.H = img_size // patch_size
        self.W = img_size // patch_size
        num_patches = self.H * self.W

        # Patch Embedding
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim))

        # Blocks
        self.blocks = nn.ModuleList([
            ThermoViTBlock(dim, heads, mlp_ratio, reg_mode, sketch_dim)
            for _ in range(depth)
        ])

        self.norm = nn.RMSNorm(dim, eps=1e-6)
        self.head = nn.Linear(dim, num_classes)

        # Initialize weights (trunc_normal is usually good for ViT)
        self.apply(self._init_weights)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B = x.shape[0]

        # Patch Embed: [B, C, H, W] -> [B, N, C]
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # Prepend CLS
        x = x + self.pos_embed

        total_phys_loss = 0.0

        # Pass through blocks
        for blk in self.blocks:
            x, l_loss = blk(x, self.H, self.W)
            total_phys_loss += l_loss

        # Classifier
        x = self.norm(x)
        out = self.head(x[:, 0])

        return out, (total_phys_loss / len(self.blocks))

def ViT():
    return ThermoViT(
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        num_classes=NUM_CLASSES,
        dim=192,
        depth=12,
        heads=3,
        mlp_ratio=4,
        reg_mode="baseline",
        sketch_dim=SKETCH_DIM
    )

# ==========================================
# 3. Data Loading
# ==========================================
class HFImageNetWrapper(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
    def __len__(self): return len(self.dataset)
    def __getitem__(self, idx):
        img = self.dataset[idx]['image']
        if img.mode != 'RGB': img = img.convert('RGB')
        return self.transform(img) if self.transform else img, self.dataset[idx]['label']

def get_val_loader(batch_size=256):
    print("==> Loading HF Dataset (Validation split)...")
    hf_ds = load_dataset("benjamin-paine/imagenet-1k-256x256", split='validation')
    val_transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    return DataLoader(HFImageNetWrapper(hf_ds, val_transform), batch_size=batch_size, shuffle=False, num_workers=4)

# ==========================================
# 4. Evaluation & Dimension Engine
# ==========================================

def compute_all_metrics(layer_acts):
    """
    Computes 4 distinct geometric metrics for a given batch of representations.
    """
    metrics = {}
    N_samples, C = layer_acts.shape
    
    # 1. Feature Norm Variance
    norms = torch.norm(layer_acts, p=2, dim=1)
    metrics['norm_variance'] = torch.var(norms).item()
    
    # Center the activations for Covariance-based metrics
    centered_acts = layer_acts - layer_acts.mean(dim=0, keepdim=True)
    cov = (centered_acts.T @ centered_acts) / (N_samples - 1)
    cov_64 = cov.to(torch.float64)
    
    # 2. Participation Ratio (Effective Dimension)
    trace_cov = torch.trace(cov_64)
    trace_cov_sq = torch.trace(cov_64 @ cov_64)
    metrics['participation_ratio'] = (trace_cov ** 2 / trace_cov_sq).item() if trace_cov_sq != 0 else 0.0
    
    # 3. Spectral Entropy (SVD Entropy)
    eigenvalues = torch.linalg.eigvalsh(cov_64)
    eigenvalues = torch.clamp(eigenvalues, min=1e-10) # Prevent log(0)
    p = eigenvalues / eigenvalues.sum()
    metrics['spectral_entropy'] = -torch.sum(p * torch.log(p)).item()
    
    # 4. Average Pairwise Cosine Similarity
    # L2 normalize the original features
    norm_acts = F.normalize(layer_acts, p=2, dim=1)
    # Cosine sim matrix: [N, N]
    sim_matrix = norm_acts @ norm_acts.T
    # Average off-diagonal elements (subtract N for the diagonal of 1s)
    avg_cos_sim = (sim_matrix.sum() - N_samples) / (N_samples * (N_samples - 1))
    metrics['avg_cosine_sim'] = avg_cos_sim.item()
    
    return metrics

@torch.no_grad()
def evaluate_model_comprehensive(model_name, weight_file, dataloader, num_ed_batches=10):
    print(f"\nEvaluating: {model_name.upper()}")
    
    weight_path = hf_hub_download(repo_id="ChavyvAkvar/sigreg-imagenet", filename=weight_file)
    net = ViT().to(DEVICE)
    
    # Clean torch.compile _orig_mod prefix
    raw_state_dict = torch.load(weight_path, map_location=DEVICE)
    clean_state_dict = {k.replace('_orig_mod.', ''): v for k, v in raw_state_dict.items()}
    net.load_state_dict(clean_state_dict)
    net.eval()
    
    activations = {i:[] for i in range(12)}
    hooks =[]
    
    def get_hook(layer_idx):
        def hook(module, input, output):
            activations[layer_idx].append(output[0].mean(dim=1).cpu())
        return hook

    for i, blk in enumerate(net.blocks):
        hooks.append(blk.register_forward_hook(get_hook(i)))
        
    for batch_idx, (inputs, _) in enumerate(dataloader):
        if batch_idx >= num_ed_batches: break
        net(inputs.to(DEVICE))
        
    for h in hooks: h.remove()
        
    # Aggregate metrics per layer
    layer_metrics = { 'PR': [], 'Entropy':[], 'CosSim': [], 'NormVar':[] }
    
    for i in range(12):
        layer_acts = torch.cat(activations[i], dim=0)
        metrics = compute_all_metrics(layer_acts)
        
        layer_metrics['PR'].append(metrics['participation_ratio'])
        layer_metrics['Entropy'].append(metrics['spectral_entropy'])
        layer_metrics['CosSim'].append(metrics['avg_cosine_sim'])
        layer_metrics['NormVar'].append(metrics['norm_variance'])
        
    return layer_metrics

# ==========================================
# Main Execution & Plotting
# ==========================================
if __name__ == '__main__':
    models_to_test = {
        'Baseline': 'vit_tiny_baseline_best.pth',
        'Strong-SIGReg': 'vit_tiny_strong_best.pth',
        'Weak-SIGReg': 'vit_tiny_weak_best.pth'
    }
    
    val_loader = get_val_loader(batch_size=256)
    results = {name: evaluate_model_comprehensive(name, file, val_loader, 100) 
               for name, file in models_to_test.items()}

    # Plotting 4 Subplots
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Comprehensive Geometric Probing of ViT Representations", fontsize=16)
    
    colors = {'Baseline': '#e74c3c', 'Weak-SIGReg': '#3498db', 'Strong-SIGReg': '#2ecc71'}
    layers = range(1, 13)

    # 1. Participation Ratio
    for name, metrics in results.items():
        axs[0, 0].plot(layers, metrics['PR'], marker='o', label=name, color=colors[name], lw=2)
    axs[0, 0].axhline(y=192, color='gray', linestyle='--', alpha=0.5)
    axs[0, 0].set_title("Effective Dimension (Participation Ratio)")
    axs[0, 0].set_ylabel("Dim (Max 192)")
    
    # 2. Spectral Entropy
    for name, metrics in results.items():
        axs[0, 1].plot(layers, metrics['Entropy'], marker='o', color=colors[name], lw=2)
    axs[0, 1].axhline(y=np.log(192), color='gray', linestyle='--', alpha=0.5) # Max entropy
    axs[0, 1].set_title("Spectral Entropy (Information Capacity)")
    axs[0, 1].set_ylabel("Shannon Entropy (Max ~5.25)")
    
    # 3. Average Cosine Similarity
    for name, metrics in results.items():
        axs[1, 0].plot(layers, metrics['CosSim'], marker='o', color=colors[name], lw=2)
    axs[1, 0].set_title("Avg Pairwise Cosine Similarity (Anisotropy)")
    axs[1, 0].set_ylabel("Similarity (0.0 = Isotropic, 1.0 = Collapsed)")
    
    # 4. Feature Norm Variance
    for name, metrics in results.items():
        axs[1, 1].plot(layers, metrics['NormVar'], marker='o', color=colors[name], lw=2)
    axs[1, 1].set_title("Feature Magnitude Variance")
    axs[1, 1].set_ylabel("Variance of L2 Norms")

    for ax in axs.flat:
        ax.set_xlabel("Block Depth")
        ax.set_xticks(layers)
        ax.grid(True, alpha=0.3)
        
    fig.legend(['Baseline', 'Strong-SIGReg', 'Weak-SIGReg', 'Theoretical Max'], 
               loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.02))
    
    plt.tight_layout()
    plt.savefig("comprehensive_geometric_probe.png", dpi=300, bbox_inches='tight')
    print("\n==> Saved comprehensive plots to 'comprehensive_geometric_probe.png'")