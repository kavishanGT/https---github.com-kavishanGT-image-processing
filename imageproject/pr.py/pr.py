import numpy as np

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Gradient descent with momentum function for two iterations
def gradient_descent(w1, w2, b, x1, x2, y, learning_rate, momentum, iterations):
    v1, v2 = 0, 0  # Initialize velocities
    for i in range(iterations):
        # Forward pass
        z = w1 * x1 + w2 * x2 + b
        y_hat = sigmoid(z)
        
        # Compute the error
        error = y - y_hat
        
        # Compute gradients
        dL_dy_hat = -(y - y_hat)
        dy_hat_dz = y_hat * (1 - y_hat)
        dz_dw1 = x1
        dz_dw2 = x2
        
        grad_w1 = dL_dy_hat * dy_hat_dz * dz_dw1
        grad_w2 = dL_dy_hat * dy_hat_dz * dz_dw2
        
        # Update velocities and weights
        v1 = momentum * v1 - learning_rate * grad_w1
        v2 = momentum * v2 - learning_rate * grad_w2
        
        w1 += v1
        w2 += v2
        
        # Print intermediate results
        print(f"Iteration {i+1}: w1 = {w1:.4f}, w2 = {w2:.4f}")
    
    return w1, w2

# Given initial values
w1, w2 = 0.5, 0.2
b = 0.1
x1, x2 = 2, 5
y = 0.3
learning_rate = 0.1
momentum = 0.9
iterations = 2

# Run gradient descent
w1_final, w2_final = gradient_descent(w1, w2, b, x1, x2, y, learning_rate, momentum, iterations)