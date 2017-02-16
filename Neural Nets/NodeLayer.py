import numpy as np
import Node

class NodeLayer(object):
    """ Node Layer - A single perceptron """

    def __init__(self, n_count, in_count):
        self.nodes = [Node.Node(in_count) for x in range(n_count)]

    # Pass in layer before
    def calculate_output(self, previous_layer):
        for node in self.nodes:
            node.calculate_output_h(previous_layer)

    def update_inner_weights(self, next_layer):
        for node in self.nodes:
            node.update_weight()

    def update_last_weights(self):
        for node in self.nodes:
            node.update_weight()