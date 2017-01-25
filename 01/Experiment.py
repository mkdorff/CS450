import sys
import numpy as np
import HardCodedClassifier as clsfr
import Dataset as ds


# change this one based off what we're getting
def load_dataset():
    # Setting up the data from csv & cleaning
    csv = np.genfromtxt('iris.csv', dtype=str, delimiter=",")
    for x in range(len(csv)):
        for y in range(len(csv[x])):
            csv[x][y] = csv[x][y].replace("\"", "")

    dataset = ds.Dataset(feature_names=csv[:1, 1:5],
                         target_names=np.array(set(csv[1:, 5:6].flatten())),
                         data=csv[1:, 1:5].astype(np.float),
                         target=csv[1:, 5:6])

    for index in range(len(dataset.target)):
        if dataset.target[index] == 'setosa':
            dataset.target[index] = 0
        elif dataset.target[index] == 'versicolor':
            dataset.target[index] = 1
        elif dataset.target[index] == 'virginica':
            dataset.target[index] = 2
    dataset.target = dataset.target.flatten().astype(np.int)

    return dataset


def randomize_dataset(dataset):
    # This syntax is off the internet, I've no idea what to call it
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
    return percentage
    # print("Predicting targets at {}% accuracy".format(percentage))


def main(argv):
    # Load
    dataset = load_dataset()

    # n-fold cross validation
    n = 10
    accuracy = 0
    for x in range(n):
        randomize_dataset(dataset)
        training_data, test_data, training_targets, test_targets = split_dataset(dataset)

        # Train, predict. Only change would be class
        classifier = clsfr.HardCodedClassifier()
        classifier.train(training_data, training_targets)
        test_results = classifier.predict(test_data)

        # How did we do?
        accuracy += report_accuracy(test_results, test_targets)

    print("With {} trials, we are achieving {}% accuracy".format(n, accuracy / n))

    return 0


if __name__ == "__main__":
    main(sys.argv)
