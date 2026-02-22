import re
import matplotlib.pyplot as plt

# ---- path to your log file ----
log_path = "log/CIFAR_100_SIGReg_Weak_R18.txt"   # change if needed

# ---- containers ----
epochs, train, phys, val, best = [], [], [], [], []

# ---- regex parser ----
pattern = re.compile(
    r"Epoch\s+(\d+).*?"
    r"Train:\s+([\d.]+).*?"
    r"Phys:\s+([\d.]+).*?"
    r"Val:\s+([\d.]+).*?"
    r"Best:\s+([\d.]+)%"
)

# ---- read + parse ----
with open(log_path, "r") as f:
    for line in f:
        m = pattern.search(line)
        if m:
            epochs.append(int(m.group(1)))
            train.append(float(m.group(2)))
            phys.append(float(m.group(3)))
            val.append(float(m.group(4)))
            best.append(float(m.group(5)))

# ---- sanity check ----
print(f"Parsed {len(epochs)} epochs")

# ---- plot losses ----
plt.figure(figsize=(10, 6))
plt.plot(epochs, train, label="Train Loss")
plt.plot(epochs, val, label="Val Loss")
plt.plot(epochs, phys, label="Phys Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Curves")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ---- plot best % ----
plt.figure(figsize=(10, 4))
plt.plot(epochs, best, label="Best (%)")
plt.xlabel("Epoch")
plt.ylabel("Best Score (%)")
plt.title("Best Validation Performance")
plt.grid(True)
plt.tight_layout()
plt.show()
