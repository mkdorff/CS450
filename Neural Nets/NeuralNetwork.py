import numpy as np
import NodeLayer as nl


class NeuralNetwork(object):
    """ Hold All The Things! Nodes & Weights & Bros """

    def __init__(self):
        self.network = []

    # pass in array, number of nodes for each layer
    def create_network(self, dataset, hidden_layers_data=[]):
        """ All nodes will be added to our network. +1 will be added for a bias node """

        # +1 for the bias nodes
        if len(hidden_layers_data) == 0:
            self.network.append(nl.NodeLayer(dataset.target_count, dataset.input_count + 1))
            return

        for x in range(len(hidden_layers_data)):
            if x == 0:
                self.network.append(nl.NodeLayer(hidden_layers_data[x], dataset.input_count + 1))
            else:
                self.network.append(nl.NodeLayer(hidden_layers_data[x], len(self.network[x - 1].nodes) + 1))

        self.network.append(nl.NodeLayer(dataset.target_count, len(self.network[len(self.network) - 1].nodes) + 1))

    # We're enforcing a bias -1 input --- Have to call this with each data item
    def compute_results(self, inputs):
        working_inputs = []
        for layer in range(len(self.network)):
            for node in self.network[layer].nodes:
                if layer == 0:
                    node.calculate_output([-1] + inputs.tolist())
                else:
                    node.calculate_output([-1] + working_inputs)

            working_inputs.clear()
            for node in self.network[layer].nodes:
                working_inputs.append(node.value)

    def fit(self, dataset):
        for row in dataset.training_data:
            self.compute_results(row)

            # for node in self.network[len(self.network) - 1].nodes:
            #     print(node.value)
            #
            # print()

    def train(self):
        # train and update weights
        return 0

    def predict(self, dataset):
        output = []
        predicted_target = []

        for row in dataset.test_data:
            self.compute_results(row)

            for node in self.network[len(self.network) - 1].nodes:
                output.append(node.value)

            predicted_target.append(output.index(max(output)))
            output.clear()

        dataset.predicted_targets = np.array(predicted_target)