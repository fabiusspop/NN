import numpy as np

inputs = np.array([[1,0],[2,0],[3,0],[0,1],[0,2]])
targets = np.array([[0],[0],[1],[0],[1]])
weights = np.array([[1],[1],[2]], dtype=float)

num_examples = np.size(inputs, 0)
num_inputs = np.size(inputs, 1) + 1
inputs = np.hstack((inputs, np.ones((num_examples, 1))))

error_history = []

def calculate_output(inputs, weights):
    return np.dot(inputs, weights) >= 0

def train(inputs, targets, weights, num_examples, num_inputs, error_history):
    error = 1
    while error != 0:
        error = 0
        for example in range(num_examples):
            output = np.dot(inputs[example, :], weights) >= 0
            error = targets[example] - output
            weights += np.reshape(inputs[example, :] * error, [num_inputs, 1])
            error_history.append(np.abs(error))
    return weights, error_history

weights, error_history = train(inputs, targets, weights, num_examples, num_inputs, error_history)

print(calculate_output(inputs, weights))
print(error_history)