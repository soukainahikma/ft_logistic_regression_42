import numpy as np
import sgd as sgd


class LogisticRegression:

    def __init__(self, lr=0.1, n_iters=1000) -> None:
        self.lr = lr
        self.n_iters = n_iters
        self.weight = []

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def fit(self, X, y):
        X_ = np.insert(X, 0, 1, axis=1)
        n_samples, n_features = X_.shape

        for i in np.unique(y):
            y_copy = np.where(y == i, 1, 0)
            w = np.ones(n_features)  # testing initialization with one
            for _ in range(self.n_iters):
                linear_pred = X_.dot(w)
                predictions = self.sigmoid(linear_pred)
                dw = ((X_.T).dot(y_copy - predictions))/n_samples
                w += self.lr * dw
            self.weight.append((w, i))

    def sgd(self, X, y):
        for i in np.unique(y):
            y_copy = np.where(y == i, 1, 0)
            w, b = sgd.mini_batch_stochastic_gradient_descent(X, y_copy, 4)
            print(w)
            w = np.insert(w, 0, b)
            self.weight.append((w, i))

    def _predict_one(self, x):
        # it's not necessar applying sigmoid function to Xθ.
        # Since we just need to take the maximum value, it's not necessary.        
        return max((x.dot(w), c) for w, c in self.weight)[1]

    def predict(self, X):
        return [self._predict_one(i) for i in np.insert(X, 0, 1, axis=1)]

    def score(self, X, y):
        return sum(self.predict(X) == y) / len(y)
