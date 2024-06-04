import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def log_loss(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def grad_log_loss(X, y_batch, y_pred):
    error = y_pred - y_batch
    gradient_bias = np.mean(error)
    gradient_weights = np.mean(error * X, axis=0)
    return gradient_weights, gradient_bias

def initialize_parameters(n_features, n_classes):
    W = np.random.randn(n_features, n_classes)
    b = np.zeros((1, n_classes))
    return W, b


def mini_batch_stochastic_gradient_descent(X, y, n_classes, batch_size=32, learning_rate=0.01, n_iterations=1000):
    n_samples, n_features = X.shape
    W, b = initialize_parameters(n_features, n_classes)

    for i in range(n_iterations):
        permutation = np.random.permutation(n_samples)
        X_shuffled = X[permutation]
        y_shuffled = y[permutation]

        for j in range(0, n_samples, batch_size):
            X_batch = X_shuffled[j:j+batch_size]
            y_batch = y_shuffled[j:j+batch_size]

            z = np.dot(X_batch, W) + b
            y_pred = sigmoid(z)
        
            dW, db = grad_log_loss(X_batch,y_batch,y_pred)

            W -= learning_rate * dW
            b -= learning_rate * db

        if i % 100 == 0:
            loss = log_loss(X, y, W, b)
            print(f"Iteration {i}, Loss: {loss}")

    return W, b