# Version for LLM

> Instead of mean pooling over tokens, we flatten batch and sequence dimension together (B*T, C)

We test two version:

- Orthogonal angle but free magnitude:
```python
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
```

- Magnitude following Zipfian rank:
```python
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
    u = x / norms # direction-only vectors
    # 4) Angular orthogonality / decorrelation loss
    # Suppress off-diagonal cosine correlations
    G = (u.T @ u) / (N - 1 + eps)
    ang_loss = torch.norm(G - torch.diag(torch.diag(G)), p='fro')
    # 5) Zipf magnitude loss
    # Zipf is rank-based, so compare sorted norms to rank^(-s)
    sorted_norms, _ = torch.sort(norms.squeeze(-1), descending=True)
    ranks = torch.arange(1, N + 1, device=x.device, dtype=x.dtype)
    zipf_target = ranks.pow(-zipf_s)
    zipf_target = zipf_target / (zipf_target.sum() + eps)
    sorted_norms = sorted_norms / (sorted_norms.sum() + eps)
    mag_loss = torch.norm(sorted_norms - zipf_target, p='fro')
    return lam_ang * ang_loss + lam_mag * mag_loss
```

Results on NanoGPT (100M params, 1B tokens fineweb-edu dataset):

<img width="2528" height="1328" alt="W B Chart 4_1_2026, 1_01_19 PM" src="https://github.com/user-attachments/assets/8eb6315d-f393-4f27-8435-e9aa04295b73" />
<img width="2528" height="1328" alt="W B Chart 4_1_2026, 1_01_30 PM" src="https://github.com/user-attachments/assets/d98f78df-0184-4b16-9c08-6c5c79c8152d" />

- Baseline (no SIGReg):
    - Train loss: 3.1379
    - Val loss: 3.1210
    - Wall time: 130262.25s (36.18 hours)
- Strong SIGReg:
    - Train loss: 7.7682
    - Val loss: 7.9404
    - Wall time: 138726.38s (38.53 hours)
- Weak SIGReg:
    - Train loss: 3.1633
    - Val loss: 3.1233
    - Wall time: 131881.89s (36.63 hours)
- Discrete SIReg:
    - Train loss: 3.1473
    - Val loss: 3.1332
    - Wall time: 135010.85s (37.50 hours)
- Zipf SIReg:
    - Train loss: 3.1302
    - Val loss: 3.1178
    - Wall time: 134855.32s (37.46 hours)
