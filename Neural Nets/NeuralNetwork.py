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

        self.training_data = standardized_data.T
        self.training_targets = training_target
        self.means = np.array(col_means)
        self.stds = np.array(col_stds)
        self.input_count = training_data.shape[1]
        self.target_count = len(np.unique(training_target))
        self.network = []


    # You don't get to not have a bias
    def create_network(self):
        meow = 0

    def fit(self, training_data, training_targets, target_count,
            hidden_layers=0, nodes_in_hidden_layer=0, bias=True):
        self.training_data = training_data
        self.training_targets = training_targets

        input_count = training_data.shape[1]

        # let's make this gross
        if hidden_layers == 0:
            node_layer = []
            for x in range(target_count):
                node_layer.append(node.Node(input_count))
            self.network.append(node_layer)

        #
        for layer in range(hidden_layers + 1):
            node_layer = []
            if layer == 0:
                for x in range(nodes_in_hidden_layer + 1):
                    node_layer.append(node.Node(input_count))
                self.network.append(node_layer)
            elif layer == hidden_layers - 1:
                for x in range(target_count):
                    node_layer.append(node.Node(len(self.network[layer - 1])))
                self.network.append(node_layer)
            else:
                for x in range(nodes_in_hidden_layer + 1):
                    node_layer.append(node.Node(len(self.network[layer-1])))
                self.network.append(node_layer)




        #
        # while hidden_layers > 0:
        #     node_layer = []
        #     for x in range(nodes_in_hidden_layer):
        #         node_layer.append(node.Node())
        #
        #
        #     hidden_layers -= 1
        #
        # # if layers > 0 we'll do stuff
        # inputs_count = training_data.shape[1]
        #
        # # this needs to be range for possible targets
        # for output in range(target_count):
        #     self.network.append(node.Node())

        # print(training_data.shape[1])
        # print(training_targets.shape)
        #
        # node.Node()
    # update network for days with nodes

    def train(self):
        # train and update weights
        return 0

    def predict(self, test_data):
        # standardize test data before predicting!
        test_results = [1 for x in range(len(test_data))]
        return np.array(test_results)
