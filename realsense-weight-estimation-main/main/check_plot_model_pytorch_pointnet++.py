import re
import matplotlib.pyplot as plt

log_text = """

=================================================================
 Starting training
=================================================================
Epoch    1/1000 | loss=0.1762  mae=0.4773 | val_loss=0.1849  val_mae=0.5141  ema=0.1849
  ✓ val_loss improved to 0.1849 → saved to /content/drive/MyDrive/cache/best_model.pt
Epoch    2/1000 | loss=0.1436  mae=0.4248 | val_loss=0.1255  val_mae=0.3947  ema=0.1730
  ✓ val_loss improved to 0.1255 → saved to /content/drive/MyDrive/cache/best_model.pt
Epoch    3/1000 | loss=0.1250  mae=0.3951 | val_loss=0.1587  val_mae=0.4649  ema=0.1702
Epoch    4/1000 | loss=0.1219  mae=0.3893 | val_loss=0.1542  val_mae=0.4567  ema=0.1670
Epoch    5/1000 | loss=0.1187  mae=0.3845 | val_loss=0.2648  val_mae=0.6311  ema=0.1865

"""

epochs = []
train_loss = []
val_loss = []
train_mae = []
val_mae = []

pattern = r"Epoch\s+(\d+)/\d+\s+\|\s+loss=([\d\.]+)\s+mae=([\d\.]+)\s+\|\s+val_loss=([\d\.]+)\s+val_mae=([\d\.]+)"

matches = re.findall(pattern, log_text)

for match in matches:
    epochs.append(int(match[0]))
    train_loss.append(float(match[1]))
    train_mae.append(float(match[2]))
    val_loss.append(float(match[3]))
    val_mae.append(float(match[4]))

print("Epochs:", epochs)
print("Train Loss:", train_loss)
print("Val Loss:", val_loss)

# ===== Plot LOSS =====
plt.figure(figsize=(8, 5))
plt.plot(epochs, train_loss, marker='o', label='Train Loss')
plt.plot(epochs, val_loss, marker='o', label='Val Loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.legend()
plt.grid(True)

# ===== Plot MAE =====
plt.figure(figsize=(8, 5))
plt.plot(epochs, train_mae, marker='o', label='Train MAE')
plt.plot(epochs, val_mae, marker='o', label='Val MAE')

plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.title('MAE over Epochs')
plt.legend()
plt.grid(True)

plt.show()