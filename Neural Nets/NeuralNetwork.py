import numpy as np
import Node as node

class NeuralNetwork(object):
    """ Hold All The Things! Nodes & Weights & Bros """

    def __init__(self, training_data, training_target):
        # print("status uuouohashdf")

        # we're going to standardize our data here
        standardized_data = training_data.T
        col_means = []
        col_stds = []

        for x in range(len(standardized_data)):
            col_means.append(np.mean(standardized_data[x]))
            col_stds.append(np.std(standardized_data[x]))
            standardized_data[x] = [(el - col_means[x]) / col_stds[x] for el in standardized_data[x]]

        # print(standardized_data)
        # print([-1 for x in range(len(standardized_data[0]))])

        self.training_data = standardized_data.T
        self.training_targets = training_target
        self.means = np.array(col_means)
        self.stds = np.array(col_stds)
        self.input_count = training_data.shape[1]
        self.target_count = len(np.unique(training_target))
        # self.network = [] .append for when we need multiple perceptrons

    # This will be more complicated the next time
    def create_network(self):
        """ All nodes will be added to our network. +1 will be added for a bias node """
        self.network = [node.Node(self.input_count + 1) for x in range(self.target_count)]

    # This is just here to make sure the mechanisms works
    def feed(self):
        for row in range(self.training_data.shape[0]):
            print("testing row {}".format(row))
            for node in range(len(self.network)):
                print(self.network[node].calculate_output_h([-1] + self.training_data[row].tolist()))

        # Looks good!

    def train(self):
        # train and update weights
        return 0

    def predict(self, test_data):
        # standardize test data before predicting!
        test_results = [1 for x in range(len(test_data))]
        return np.array(test_results)
