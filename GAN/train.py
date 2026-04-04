import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time

# ==========================================
# 1. Configuration
# ==========================================
# OPTIONS: 'baseline', 'weak' (Covariance), 'strong' (LeJEPA CF)
REG_MODE = 'strong'  
SIGR_ALPHA = 0.1   # Strength of the SIGReg constraint
SKETCH_DIM = 64    # Dimension of the random observer

BATCH_SIZE = 256
Z_DIM = 128
LEARNING_RATE = 2e-4
EPOCHS = 400
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if torch.backends.mps.is_available(): DEVICE = 'mps'

print(f"Training on: {DEVICE} | Mode: {REG_MODE} | Alpha: {SIGR_ALPHA}")

# ==========================================
# 2. Physics Engine: The Regularizers
# ==========================================

def sigreg_weak_loss(x, sketch_dim=64):
    """
    Weak-SIGReg: Forces Covariance(x) ~ Identity via Sketching.
    Prevents dimensional collapse in the representations.
    """
    N, C = x.size()
    if C > sketch_dim:
        S = torch.randn(sketch_dim, C, device=x.device) / (C ** 0.5)
        x = x @ S.T
    else:
        sketch_dim = C

    # Centering & Covariance
    x = x - x.mean(dim=0, keepdim=True)
    cov = (x.T @ x) / (N - 1 + 1e-6)

    # Target Identity
    target = torch.eye(sketch_dim, device=x.device)

    # Minimize distance to Identity
    return torch.norm(cov - target, p='fro')

def sigreg_strong_loss(x, sketch_dim=64):
    """
    Strong-SIGReg (LeJEPA): Forces ECF(x) ~ ECF(Gaussian).
    Matches all moments using random 1D projections.
    """
    N, C = x.size()
    if C > sketch_dim:
        A = torch.randn(C, sketch_dim, device=x.device)
    else:
        A = torch.randn(C, C, device=x.device)
        
    A = A / (A.norm(p=2, dim=0, keepdim=True) + 1e-6)

    t = torch.linspace(-5, 5, 17, device=x.device)
    exp_f = torch.exp(-0.5 * t**2)

    proj = x @ A
    args = proj.unsqueeze(2) * t.view(1, 1, -1)
    
    ecf = torch.exp(1j * args).mean(dim=0)
    
    diff_sq = (ecf - exp_f.unsqueeze(0)).abs().square()
    err = diff_sq * exp_f.unsqueeze(0)
    
    loss = torch.trapz(err, t, dim=1) * N
    return loss.mean()

def compute_multi_layer_sigreg(features_list):
    """Computes the average SIGReg loss across multiple layers."""
    if REG_MODE == 'baseline':
        return torch.tensor(0.0, device=DEVICE)
    
    total_reg = 0.0
    for feat in features_list:
        if REG_MODE == 'weak':
            total_reg += sigreg_weak_loss(feat, SKETCH_DIM)
        elif REG_MODE == 'strong':
            total_reg += sigreg_strong_loss(feat, SKETCH_DIM)
            
    return total_reg / len(features_list)

# ==========================================
# 3. The 25-Gaussians Dataset
# ==========================================
def get_25_gaussians_batch(batch_size):
    """Generates a batch of points from a 5x5 grid of Gaussians."""
    centers =[]
    for x in range(-2, 3):
        for y in range(-2, 3):
            centers.append([x * 2.0, y * 2.0])
    centers = np.array(centers)
    
    indices = np.random.choice(25, batch_size)
    batch_centers = centers[indices]
    
    noise = np.random.normal(0, 0.05, size=(batch_size, 2))
    points = batch_centers + noise
    return torch.tensor(points, dtype=torch.float32)

# ==========================================
# 4. Models: Multi-Layer Feature Extraction
# ==========================================

class Generator(nn.Module):
    def __init__(self, z_dim, out_dim=2):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(z_dim, 512)
        self.act1 = nn.GELU()
        self.fc2 = nn.Linear(512, 512)
        self.act2 = nn.GELU()
        self.fc3 = nn.Linear(512, 512)
        self.act3 = nn.GELU()
        self.fc4 = nn.Linear(512, 512)
        self.act4 = nn.GELU()
        self.fc5 = nn.Linear(512, 512)
        self.act5 = nn.GELU()
        self.fc6 = nn.Linear(512, 512)
        self.act6 = nn.GELU()
        self.out_layer = nn.Linear(512, out_dim)

    def forward(self, x):
        # Extract features at every layer
        f1 = self.act1(self.fc1(x))
        f2 = self.act2(self.fc2(f1))
        f3 = self.act3(self.fc3(f2))
        f4 = self.act1(self.fc4(f3))
        f5 = self.act2(self.fc5(f4))
        f6 = self.act3(self.fc6(f5))
        out = self.out_layer(f6)
        # We do NOT regularize the final output, as it shouldn't be an isotropic Gaussian
        return out, [f1, f2, f3, f4, f5, f6]

class Discriminator(nn.Module):
    def __init__(self, in_dim=2):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(in_dim, 128)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        self.fc2 = nn.Linear(128, 128)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)
        self.fc3 = nn.Linear(128, 128)
        self.act3 = nn.LeakyReLU(0.2, inplace=True)
        self.logits = nn.Linear(128, 1)

    def forward(self, x):
        # Extract features at every layer
        f1 = self.act1(self.fc1(x))
        f2 = self.act2(self.fc2(f1))
        f3 = self.act3(self.fc3(f2))
        out = self.logits(f3)
        # We do NOT regularize the 1D logits
        return out,[f1, f2, f3]

# ==========================================
# 5. Training Loop
# ==========================================
def train():
    G = Generator(Z_DIM).to(DEVICE)
    D = Discriminator().to(DEVICE)

    opt_G = optim.Adam(G.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()

    print("Starting Training...")
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        for _ in range(100):
            real_data = get_25_gaussians_batch(BATCH_SIZE).to(DEVICE)
            
            # ==============================
            #  1. Train Discriminator
            # ==============================
            opt_D.zero_grad()
            
            real_logits, d_real_features = D(real_data)
            loss_d_real = criterion(real_logits, torch.ones_like(real_logits))
            
            z = torch.randn(BATCH_SIZE, Z_DIM, device=DEVICE)
            fake_data_detached, _ = G(z) 
            fake_data_detached = fake_data_detached.detach()
            
            fake_logits, _ = D(fake_data_detached)
            loss_d_fake = criterion(fake_logits, torch.zeros_like(fake_logits))
            
            # --- Discriminator SIGReg (Applied to all D hidden layers) ---
            d_reg_loss = compute_multi_layer_sigreg(d_real_features)

            loss_d = loss_d_real + loss_d_fake + (SIGR_ALPHA * d_reg_loss)
            loss_d.backward()
            opt_D.step()

            # ==============================
            #  2. Train Generator
            # ==============================
            opt_G.zero_grad()
            
            z = torch.randn(BATCH_SIZE, Z_DIM, device=DEVICE)
            fake_data, g_features = G(z)
            fake_logits_for_G, _ = D(fake_data)
            
            loss_g_adv = criterion(fake_logits_for_G, torch.ones_like(fake_logits_for_G))
            
            # --- Generator SIGReg (Applied to all G hidden layers) ---
            g_reg_loss = compute_multi_layer_sigreg(g_features)

            loss_g = loss_g_adv + (SIGR_ALPHA * g_reg_loss * 0.1)
            loss_g.backward()
            opt_G.step()

        # Logging
        if epoch % 50 == 0 or epoch == 1:
            print(f"Epoch[{epoch:03d}/{EPOCHS}] | D Loss: {loss_d.item():.4f} (Reg: {d_reg_loss.item():.4f}) | G Loss: {loss_g.item():.4f} (Reg: {g_reg_loss.item():.4f})")

    print(f"Training finished in {time.time() - start_time:.0f} seconds.")
    return G

# ==========================================
# 6. Evaluation / Plotting
# ==========================================
def evaluate_and_plot(G):
    G.eval()
    
    with torch.no_grad():
        z = torch.randn(2500, Z_DIM, device=DEVICE)
        generated_points, _ = G(z)
        generated_points = generated_points.cpu().numpy()
        
    real_points = get_25_gaussians_batch(2500).numpy()

    plt.figure(figsize=(8, 8))
    plt.scatter(real_points[:, 0], real_points[:, 1], c='blue', alpha=0.3, label='Real Data', s=10)
    plt.scatter(generated_points[:, 0], generated_points[:, 1], c='red', alpha=0.3, label='Generated (Fake)', s=10)
    
    plt.xlim(-6, 6)
    plt.ylim(-6, 6)
    plt.title(f"GAN on 25 Gaussians \nMode: {REG_MODE.upper()} | Alpha: {SIGR_ALPHA}")
    plt.legend()
    
    filename = f"GAN_{REG_MODE}.png"
    plt.savefig(filename, dpi=150)
    print(f"Result saved to {filename}")
    plt.show()

if __name__ == '__main__':
    trained_generator = train()
    evaluate_and_plot(trained_generator)