import numpy as np


class Node(object):
    """ Node and Co. """

    def __init__(self, w_counts):
        self.value = 0
        self.weights = [np.random.uniform(-1.0) for x in range(w_counts)]
        self.past_weight = 0

    def calculate_output(self, inputs):
        """ Uses a sigmoid function - f(x) = 1 / (1 + e^-x) """
        if len(inputs) != len(self.weights):
            raise IndexError("The number of weights does not match the number of inputs")

        self.value = 1 / (1 + np.e**(-np.dot(self.weights, inputs)))

    def update_weight(self):
        return 0