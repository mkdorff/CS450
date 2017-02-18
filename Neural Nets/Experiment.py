import sys
import numpy as np
import NeuralNetwork as net
import Dataset as ds


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
    neural_net.create_network(dataset, [2, 3])
    neural_net.fit(dataset)
    neural_net.predict(dataset)

    # How did we do?
    dataset.report_accuracy()

    return 0

if __name__ == "__main__":
    main(sys.argv)
