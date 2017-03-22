import sys
import Dataset as ds
import kNN as knn
from sklearn import svm


def main(argv):
    # Load, randomize, set
    dataset = ds.Dataset()
    dataset.load_lung_cancer()

    print(dataset.target)
    print(dataset.data)
    dataset.randomize_data()
    dataset.split_data()
    dataset.standardize_data()

    # Train, predict
    neural_net = net.NeuralNetwork()
    #
    # # Optional array for hidden nodes. Putting a number in the array means we want a
    # # hidden layer there. The number we put in is the number of nodes in that layer
    # # neural_net.create_network(dataset)
    neural_net.create_network(dataset, [2, 2, 2])
    #
    # # We just need to edit these values :D
    # error_data = neural_net.fit(dataset, learning_rate=0.4, momentum=0.9,
    #                             error_change_percent=0.01, epoch_iterations=200)
    # neural_net.predict(dataset)
    #
    # # How did we do?
    # dataset.report_accuracy()

    return 0


if __name__ == "__main__":
    main(sys.argv)
