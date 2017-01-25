import numpy as np


class HardCodedClassifier(object):
    """ A basic classifier that doesn't doesn't do too much """

    def __init__(self):
        self.algorithm_data = []
        # I don't really know what this will look like

    def train(self, training_data, training_targets):
        # This is obnoxious and I don't want to see this
        # print("Now training...")
        return 0

    def predict(self, test_data):
        test_results = [1 for x in range(len(test_data))]
        return np.array(test_results)
