# source: https://www.youtube.com/watch?v=tMrbN67U9d4

import numpy as np

inputs = [1, 2, 3, 2.5]
weights = [
    [0.2, 0.8, -0.5, 1.0],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]
]

biases = [
    2,
    3,
    0.5
]

# layer_outputs = []

# for weights, biases in zip(weights, biases):
#     neuron_output = 0

#     for n_input, weight in zip(inputs, weights):
#         neuron_output += n_input*weight
#     neuron_output += biases
#     layer_outputs.append(neuron_output)

# print(layer_outputs)

"""
Calculating output using dot product

a = [1, 2, 3]
b = [2, 3, 4]

a.b = [1,2,3].[2,3,4] = 1*2 + 2*3 + 3*4 = 20

Dot product of weights & inputs:

inputs = [1, 2, 3, 4]
weights = [
    [1, 2, 3, 4],
    [1, 2, 3, 4],
    [1, 2, 3, 4],
]

Shape of dot product:
Input = (4,0)
Weights = (3, 4)
Output = (3,0) ... (num columns in weights, num rows in inputs)

weights.inputs = [[weights].[inputs]] = [
    weight[1]*input[1] + ... + weight[n] + input[n] = x
    1*1 + 2*2 + 3*3, 4*4 = 30,
    1*1 + 2*2 + 3*3, 4*4 = 30,
    1*1 + 2*2 + 3*3, 4*4 = 30
]

inputs must be of equal shape to perform a dot product multiplication
"""

output = np.dot(weights, inputs) + biases

print(output) # [4.8, 1.21, 2.385]


"""
Shape:

Array:
[1, 2, 3, 4]    Shape: (4,)     Type: 1D array, Vector

[[1, 2, 3, 4],
[1, 2, 3, 4]]   Shape: (2,4)    Type: 2D array, Matrix

[
    [
        [1, 2, 3, 4],
        [1, 2, 3, 4]
    ],
    [
        [1, 2, 3, 4],
        [1, 2, 3, 4]
    ],
    [
        [1, 2, 3, 4],
        [1, 2, 3, 4]
    ]
]   Shape: (3, 2, 4)    Type: 3D array

Tensor: An object that can be represented at an Array
"""