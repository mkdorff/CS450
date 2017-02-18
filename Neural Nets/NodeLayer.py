import Node


class NodeLayer(object):
    """ Node Layer - A single perceptron """

    def __init__(self, n_count, in_count):
        self.nodes = [Node.Node(in_count) for x in range(n_count)]

    def update_inner_weights(self, next_layer):
        for node in self.nodes:
            node.update_weight()

    def update_last_weights(self):
        for node in self.nodes:
            node.update_weight()
