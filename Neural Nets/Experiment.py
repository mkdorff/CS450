import sys
import numpy as np
import NeuralNetwork as clsfr
import Dataset as ds


def randomize_dataset(dataset):
    # This syntax is off the internet, I've no idea what the sytnax
    reorder = np.random.permutation(len(dataset.data))
    dataset.data = dataset.data[reorder]
    dataset.target = dataset.target[reorder]


def split_dataset(dataset):
    # We're doing 70/30 here
    training_size = round(len(dataset.data) * .7)
    training_data, test_data = np.split(dataset.data, [training_size])
    training_targets, test_targets = np.split(dataset.target, [training_size])
    # Is this ugly? I don't know. Probably.
    return training_data, test_data, training_targets, test_targets


def report_accuracy(test_results, test_targets):
    correct = 0
    for i in range(len(test_results)):
        if test_results[i] == test_targets[i]:
            correct += 1
    percentage = round(correct / len(test_results), 2) * 100
    print("Predicting targets at {}% accuracy".format(percentage))


def main(argv):
    # Load
    dataset = ds.Dataset()
    # dataset.load_dataset_from_iris_csv('iris.csv')
    dataset.load_from_txts_if_numerical('iris.names.txt', 'iris.data.txt')

    randomize_dataset(dataset)
    training_data, test_data, training_targets, test_targets = split_dataset(dataset)

    # Train, predict
    classifier = clsfr.NeuralNetwork(training_data, training_targets)
    # classifier.fit(training_data, training_targets, dataset.target_count)

    # print(classifier.network)
    # print(classifier.network[0][1].weights)
    # print(classifier.network[0][2].weights)


    # print(classifier.training_data)
    # print(classifier.training_targets)

    # How did we do?
    # report_accuracy(test_results, test_targets)

    # # Playground
    # try:
    #     nerrrds = node.Node([1, 2, 3], [0.2, -0.4, 0.5])
    # except IndexError as e:
    #     print(e)

    # meow = [[]]
    # meow[0] = [1,2,3]
    # meow[1] = [1, 2, 3]

    # print(meow)

    return 0


if __name__ == "__main__":
    main(sys.argv)
