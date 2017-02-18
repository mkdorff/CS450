import sys
import numpy as np
import NeuralNetwork as net
import Dataset as ds

import Node

def report_accuracy(test_results, test_targets):
    correct = 0
    for i in range(len(test_results)):
        if test_results[i] == test_targets[i]:
            correct += 1
    percentage = round(correct / len(test_results), 2) * 100
    print("Predicting targets at {}% accuracy".format(percentage))


def main(argv):

    # Load, randomize, set
    dataset = ds.Dataset()
    dataset.load_from_txts('iris.names.txt', 'iris.data.txt')
    # dataset.load_from_txts('pima-indians-diabetes.names.txt', 'pima-indians-diabetes.data.txt')
    dataset.randomize_data()
    dataset.split_data()
    dataset.standardize_data()

    # Train, predict
    neural_net = net.NeuralNetwork()
    # neural_net.create_network(dataset)
    # neural_net.compute_results(dataset.training_data[0])
    # neural_net.fit(dataset)

    # print(dataset.input_count)
    # print(dataset.target_count)
    # print(neural_net.network[0].nodes[0].weights)
    # neural_net.network[0].nodes[0].calculate_output([-1,2,3,4,5,6,7,8,9])
    # Example
    # neural_net.create_network(dataset, [4])
    # neural_net.fit(dataset)
    # neural_net.compute_results(dataset.training_data[0])
    neural_net.create_network(dataset, [2, 3])
    neural_net.create_network(dataset, [5, 3, 9])
    neural_net.fit(dataset)
    # print(neural_net.network[0].nodes[0].weights)
    # print(neural_net.network[1].nodes[0].weights)

    # print([-1] + [2,3,4])

    # classifier = clsfr.NeuralNetwork(training_data, training_targets)
    # classifier.create_network()
    # classifier.feed()

    # How did we do?
    # report_accuracy(test_results, test_targets)

    # node = Node.Node(4)
    # print(node.weights)

    # We can make

    return 0


if __name__ == "__main__":
    main(sys.argv)
