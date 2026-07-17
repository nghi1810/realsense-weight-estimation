import numpy as np
import matplotlib.pyplot as plt

data = np.load(r"C:\Users\phand\Desktop\New folder\dataset\depth\apple_0.npy")

plt.imshow(data, cmap="jet")
plt.colorbar()
plt.title("Depth range")
plt.show()