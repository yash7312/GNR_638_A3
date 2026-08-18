import pickle, os
import matplotlib.pyplot as plt

with open("checkpoints/train_logs_dataset_2.pkl", "rb") as f:
    logs = pickle.load(f)

losses = logs["losses"]
accs = logs["accs"]
epochs = range(1, len(losses)+1)

os.makedirs("plots_dataset_2", exist_ok=True)

plt.figure()
plt.plot(epochs, losses, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Dataset-2 Training Loss")
plt.grid(True)
plt.savefig("plots_dataset_2/loss.png")
plt.close()

plt.figure()
plt.plot(epochs, accs, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Dataset-2 Training Accuracy")
plt.grid(True)
plt.savefig("plots_dataset_2/accuracy.png")
plt.close()

print("Plots generated successfully.")
