import numpy as np


class Node(object):
    """ Node and Co. """

    def __init__(self, input_count):
        self.input_count = input_count
        self.weights = [np.random.uniform(-1.0) for x in range(input_count)]
        self.output = 0
        self.old_output = 0

    def calculate_output_h(self, inputs):
        """ h function returns 0 or 1 """
        if len(inputs) != self.input_count:
            raise IndexError("The number of weights does not match the number of inputs")

        return 1 if np.dot(self.weights, inputs) > 0 else 0

    def calculate_output_sigmoid(self):
        """ returns sigmoid for multi-layer stuff """
        return 0
