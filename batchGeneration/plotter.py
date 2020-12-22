import numpy as np
import matplotlib.pyplot as plt

# For plotting batches

batch = np.load("batch.npy")

print(batch.shape)

for i in range(batch.shape[0]):
    plt.plot(batch[i])

plt.savefig("batch.png")