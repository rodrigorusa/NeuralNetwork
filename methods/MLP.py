import numpy as np


class MLP(object):
    @staticmethod
    def sigmoid(x, derive=False):
        if derive:
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def tahn(x, derive=False):
        if derive:
            return 1. - x * x
        return np.tanh(x)

    @staticmethod
    def relu(x, derive=False):
        if derive:
            return 1. * (x > 0)
        return x * (x > 0)

    # TODO
    @staticmethod
    def leaky_relu(x, derive=False):
        if derive:
            if x > 0:
                return 1. * (x > 0)
            else:
                return 0.01 * (x <= 0)
        else:
            if x > 0:
                return x
            else:
                return 0.01*x

    @staticmethod
    def forward(x, w, g):
        a = [g(np.dot(x, w[0]))]
        for i in range(1, len(w)):
            a.append(g(np.dot(a[i-1], w[i])))
        return a

    @staticmethod
    def backward(x, y, a, w, g):
        n = len(w)
        delta = []
        for i in range(0, n):
            delta.append([])

        delta_a_o_error = a[n-1] - y
        delta_z_o = g(a[n-1], derive=True)
        multiply = delta_a_o_error * delta_z_o
        delta[n-1] = np.dot(a[n-2].T, multiply)
        for i in range(n-2, 0, -1):
            delta_a_h = np.dot(multiply, w[i+1].T)
            delta_z_h = MLP.sigmoid(a[i], derive=True)
            multiply = delta_a_h * delta_z_h
            delta[i] = np.dot(a[i-1].T, multiply)

        delta_a_h = np.dot(multiply, w[1].T)
        delta_z_h = g(a[0], derive=True)
        delta[0] = np.dot(x.T, delta_a_h * delta_z_h)

        return delta

    @staticmethod
    def update(w, delta, eta):
        w[0] = w[0] - eta * delta[0]
        for i in range(1, len(w)):
            w[i] = w[i] - eta * delta[i]

    @staticmethod
    def build_model(x, y, layers, activation, epsilon, eta, epochs):

        # Add bias to input (x0)
        layers[0] += 1
        X = np.ones((x.shape[0], x.shape[1] + 1))
        X[:, :-1] = x

        # Reshape the label
        Y = y.reshape((y.shape[0], 1))

        # Set activation function
        g = None
        if activation == 'sigmoid':
            g = MLP.sigmoid
        elif activation == 'tahn':
            g = MLP.tahn
        elif activation == 'relu':
            g = MLP.relu
        elif activation == 'leaky-relu':
            g = MLP.leaky_relu

        w = []
        model = {}
        n_layers = len(layers)

        # Initialize the weights with random numbers
        for i in range(0, n_layers-1):
            w.append(np.random.randn(layers[i], layers[i+1]) * (2 * epsilon) - epsilon)

        # For each epoch
        for epoch in range(epochs):

            # Feed forward
            a = MLP.forward(X, w, g)

            # Calculate the error
            loss = np.mean(0.5 * np.power((a[1] - Y), 2))
            print('Epoch: %d' % (epoch+1), 'Loss: %f' % loss)

            # Backpropagation
            delta = MLP.backward(X, Y, a, w, g)

            # Update the weights
            MLP.update(w, delta, eta)

        model['w'] = w
        model['activation'] = g

        print('Coefficients: ', model['w'])

        #a = MLP.forward(X, w, g)
        #print(a[len(w)-1])

        return model
