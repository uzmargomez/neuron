# Exponential in series definition
def expuz(x, n=10000000):
    return (1.+x/n)**n

# Sigmoid function definition
def sigmoid(x):
    return 1./(1.+expuz(-x))

# Input
x = [0.2, 0.1, 0.3]

# Weights
w = [0.25, 0.2, 0.4]

# Training step
coef = 1

# Desired output
y = 0.3

# Iterations
n=2000000

for t in range(n):
    xw1 = 0.0

    # Dot product (in a general case, w will be a matrix)
    for i in range(len(x)):
        xw1 += x[i]*w[i]

    # Activation
    o = sigmoid(xw1)

    grad = []

    # Computing radient
    for i in range(len(x)):
        grad.append((o-y)*o*(1.-o)*x[i])

    # Weight update using gradient descend
    for i in range(len(x)):
        w[i] -= coef*grad[i]

print("Predicted value after", n, "iterations:" , o)
