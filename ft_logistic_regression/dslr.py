import numpy as np
import optimization_algo as opt
import pickle


class LogisticRegression:

    def __init__(self, lr=0.1, n_iters=1000) -> None:
        self.lr = lr
        self.n_iters = n_iters
        self.weight = []

    def gd(self, X, y):
        X_ = np.insert(X, 0, 1, axis=1)

        for i in np.unique(y):
            y_encoded = np.where(y == i, 1, 0)
            w = opt.gradient_descent(X_, y_encoded)
            self.weight.append((w, i))
        self.save('gd_model.pkl')

    def mb_sgd(self, X, y):
        for i in np.unique(y):
            y_encoded = np.where(y == i, 1, 0)
            w, b = opt.mini_batch_stochastic_gradient_descent(X, y_encoded, 4)
            w = np.insert(w, 0, b)
            self.weight.append((w, i))
        self.save('mb_sgd_model.pkl')


    def sgd(self, X, y):
        for i in np.unique(y):
            y_encoded = np.where(y == i, 1, 0)
            w, b = opt.stochastic_gradient_descent(X, y_encoded, 4)
            w = np.insert(w, 0, b)
            self.weight.append((w, i))
        self.save('sgd_model.pkl')

    def save(self,path):
        try:
            with open(path, 'wb') as f:
                pickle.dump((self.weight), f)
        except ValueError:
            AssertionError('Error while saving model.txt')

    def get_file_info(self, path):
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception as error:
            raise error

    def _predict_one(self, x, weight):
        # it's not necessar applying sigmoid function to Xθ.
        # Since we just need to take the maximum value, it's not necessary.
        return max((x.dot(w), c) for w, c in weight)[1]

    def predict(self, X, path):
        weight = self.get_file_info(path)
        return [self._predict_one(i, weight)
                for i in np.insert(X, 0, 1, axis=1)]

    def estimation(self, X):
        return [self._predict_one(i, self.weight)
                for i in np.insert(X, 0, 1, axis=1)]

    def score(self, X, y):
        return sum(self.predict(X) == y) / len(y)
