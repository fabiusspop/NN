import numpy as np

input_data = np.array([[1,0],[2,0],[3,0],[0,1],[0,2]])
target_output = np.array([[0], [0], [1], [0], [1]])

num_examples = np.size(input_data, 0)
input_data = np.concatenate((input_data, np.ones([num_examples, 1])), axis=1)

num_inputs = np.size(input_data, 1)
weights = np.array([[1], [1], [2]])

error_gradient = 1
while error_gradient > 0:
    # Calculate network output
    net_output = np.dot(input_data, weights)
    network_output = net_output >= 0

    # Update weights
    error = target_output - network_output
    error_gradient = sum(abs(error))
    delta_weights = error * input_data
    delta_weights = np.sum(delta_weights, axis=0)
    delta_weights = np.reshape(delta_weights, [num_inputs, 1])
    weights = weights + delta_weights


print("Final weights:\n", weights)
print("Final network output:\n", network_output)