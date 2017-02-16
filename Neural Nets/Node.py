import numpy as np


class Node(object):
    """ Node and Co. """

    def __init__(self, w_counts):
        self.value = np.random.random
        self.weights = [np.random.uniform(-1.0) for x in range(w_counts)]
        self.past_weight = 0
        # self.weights = [np.random.uniform(-1.0) for x in range(input_count)]
        self.output = 0


    def calculate_output_h(self, inputs):
        """ h function returns 0 or 1 """
        if len(inputs) != len(self.weights):
            raise IndexError("The number of weights does not match the number of inputs")

        self.output = 1 if np.dot(self.weights, inputs) > 0 else 0
        # return self.output

    def calculate_output_sigmoid(self):
        """ returns sigmoid for multi-layer stuff """
        return 0

    def update_weight(self):
        return 0