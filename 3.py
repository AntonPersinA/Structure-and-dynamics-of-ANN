import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np


class HopfieldNetwork:
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons
        self.weights = np.zeros((num_neurons, num_neurons))

    def train(self, patterns):
        for pattern in patterns:
            print(pattern)
            self.weights += np.outer(pattern, pattern)
        # Ensure no self-connections (diagonal should be zero)
        np.fill_diagonal(self.weights, 0)
        print(self.weights)
        self.weights /= patterns.shape[0]

    def predict(self, input_pattern, steps=10):
        pattern = input_pattern.copy()
        for _ in range(steps):
            for i in range(self.num_neurons):
                # Update each neuron
                pattern = np.where(np.dot(self.weights, pattern) >= 0, 1, -1)
        return pattern


x1 = [[1, 1, 0],
      [1, 0, 0],
      [1, 0, 0]]
x2 = [[0, 0, 1],
      [0, 0, 1],
      [0, 1, 1]]
y1 = [[0, 1, 0],
      [0, 1, 0],
      [0, 1, 0]]

train = [x1, x2]
test = [y1]

train = np.array(train) * 2 - 1
test = np.array(test) * 2 - 1

train = train.reshape(train.shape[0], -1)
test = test.reshape(test.shape[0], -1)

model_H = HopfieldNetwork(train.shape[1])
model_H.train(train)


steps = 3
fig, axes = plt.subplots(test.shape[0], steps + 1, figsize=(3 * steps, 6))

axes[0].imshow(test[0].reshape(3, 3))
axes[0].set_title("input")
axes[0].axis('off')

output = test[0]
for j in range(1, steps + 1):
    output = model_H.predict(output, steps=1)
    img_output = (output + 1) / 2
    axes[j].imshow(img_output.reshape((3, 3)))
    axes[j].set_title(f"n={j}")
    axes[j].axis('off')

plt.tight_layout()
plt.show()
