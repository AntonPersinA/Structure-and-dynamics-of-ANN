import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np


class HopfieldNetwork:
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons
        self.weights = np.zeros((num_neurons, num_neurons))

    def train(self, patterns):
        for pattern in patterns:
            self.weights += np.outer(pattern, pattern)
        # Ensure no self-connections (diagonal should be zero)
        np.fill_diagonal(self.weights, 0)
        self.weights /= patterns.shape[0]

    def predict(self, input_pattern, steps=10):
        pattern = input_pattern.copy()
        for _ in range(steps):
            for i in range(self.num_neurons):
                # Update each neuron
                pattern = np.where(np.dot(self.weights, pattern) >= 0, 1, -1)
        return pattern

size = np.array([3, 3])
train = np.where(np.random.random((500, np.prod(size))) > 0.5, 1, -1)

model_H = HopfieldNetwork(train.shape[1])
model_H.train(train)
model_H.weights = np.abs(model_H.weights)

c = 5
for one_count in range(int(np.prod(size) / 2) - 3, int(np.prod(size) / 2) + 3, 1):
    el = -np.ones((np.prod(size), ))
    for i in range(one_count):
        el[i] = 1

    fig, axes = plt.subplots(c, 2, figsize=(6, 3 * c))
    for i in range(c):
        np.random.shuffle(el)
        axes[i, 0].imshow((el.reshape(size) + 1) / 2, cmap='viridis', vmin=0, vmax=1)

        output = model_H.predict(el, steps=2)
        axes[i, 1].imshow((output.reshape(size) + 1) / 2, cmap='viridis', vmin=0, vmax=1)

        axes[i, 0].set_title("input")
        axes[i, 0].axis('off')
        axes[i, 1].set_title("output")
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.show()


