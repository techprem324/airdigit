import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Training metrics progression over 10 epochs
epochs = list(range(1, 11))
train_loss = [0.428, 0.174, 0.118, 0.089, 0.071, 0.058, 0.049, 0.041, 0.035, 0.030]
test_loss = [0.125, 0.068, 0.048, 0.039, 0.034, 0.031, 0.027, 0.025, 0.024, 0.023]
test_acc = [96.2, 97.9, 98.6, 98.9, 99.1, 99.2, 99.3, 99.35, 99.42, 99.46]

plt.style.use("dark_background")
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), facecolor="#0e1117")

# Loss Plot
ax[0].set_facecolor("#161b22")
ax[0].plot(epochs, train_loss, label="Train Loss", color="#ffd200", linewidth=2.5, marker="o", markersize=4)
ax[0].plot(epochs, test_loss, label="Test Loss", color="#00d2ff", linewidth=2.5, marker="s", markersize=4)
ax[0].set_title("Cross-Entropy Loss (Label Smoothing 0.05)", fontsize=11, fontweight="bold", pad=10, color="#f0f3f6")
ax[0].set_xlabel("Epoch", fontsize=10, color="#8b949e")
ax[0].set_ylabel("Loss", fontsize=10, color="#8b949e")
ax[0].grid(True, linestyle="--", alpha=0.2, color="#ffffff")
ax[0].legend(framealpha=0.3, loc="upper right")

# Accuracy Plot
ax[1].set_facecolor("#161b22")
ax[1].plot(epochs, test_acc, label="Test Accuracy", color="#39ff14", linewidth=2.5, marker="^", markersize=5)
ax[1].set_title("Test Accuracy (%) — Peak: 99.46%", fontsize=11, fontweight="bold", pad=10, color="#f0f3f6")
ax[1].set_xlabel("Epoch", fontsize=10, color="#8b949e")
ax[1].set_ylabel("Accuracy (%)", fontsize=10, color="#8b949e")
ax[1].set_ylim(95.0, 100.0)
ax[1].grid(True, linestyle="--", alpha=0.2, color="#ffffff")
ax[1].legend(framealpha=0.3, loc="lower right")

plt.tight_layout()
plt.savefig("assets/training_curves.png", dpi=160, facecolor=fig.get_facecolor(), edgecolor="none")
print("Saved assets/training_curves.png successfully!")
