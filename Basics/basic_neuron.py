# source: https://www.youtube.com/watch?v=Wo5dMEP_BbI

# Input parameters
inputs = [
    1.2,
    5.1,
    2.1
]

# Random initialized weights
weights = [
    3.1,
    2.1,
    8.7
]

bias = 3

# y = w1*x1 + w2*x2 ... wn*xn + bias
output = (inputs[0] * weights[0] +
        inputs[1] * weights[1] +
        inputs[2] * weights[2] + bias)

print(output) # 35.7