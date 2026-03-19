# Discrete SIReg for nanoGPT

We formulate discrete version of SIGReg for language model with slight modification. We normalize the magnitude of the embedding so that the regularization term only focus on the direction or angle of the embedding. Making it precisely called Sketched Isotropic Regularization (SIReg) since we don't enforce the distribution of the embedding to be Gaussian.

We test two version:

- scaling by 1/sqrt(C)
```python
def sireg_discrete_loss(x, sketch_dim=64):
    """
    Discrete-SIGReg: Forces directional orthogonality (Static Entropy) 
    while allowing magnitudes (outlier tokens) to grow freely.
    """

    N, C = x.size()

    # -----------------------------------------------------------
    # Isolate Angles from Magnitudes
    # We L2-normalize each token's embedding vector.
    # This protects the heavy-tailed norms that LLMs rely on!
    # -----------------------------------------------------------
    x = F.normalize(x, p=2, dim=-1) 

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

    # 4. Target Identity (Scaled for the normalized variance)
    target = torch.eye(sketch_dim, device=x.device) * (1.0 / C)

    # 5. Frobenius Norm
    loss = torch.norm(cov - target, p='fro')

    return loss
```

- scaling by 1
```python
def sireg_discrete_loss(x, sketch_dim=64):
    """
    Discrete-SIGReg: Forces directional orthogonality (Static Entropy) 
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

Early results on 100M tokens fineweb-edu dataset:

- Baseline (no SIGReg):
    - Train loss: 4.5648
    - Val loss: 5.1578
- scaling by 1/sqrt(C):
    - Train loss: 4.5593
    - Val loss: 5.1500
- scaling by 1:
    - Train loss: 4.5541
    - Val loss: 5.1559