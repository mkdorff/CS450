import sys
import numpy as np
import kNN as clsfr
import Dataset as ds


# IDEAS: split 3 way, add the validation thing?
# We'll work with this for now, but we'll have to read from file
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
    dataset = load_dataset()

    randomize_dataset(dataset)
    training_data, test_data, training_targets, test_targets = split_dataset(dataset)

    # Train, predict
    classifier = clsfr.kNN()
    classifier.train(training_data, training_targets)

    test_results = classifier.predict(test_data, k=8)

    # How did we do?
    report_accuracy(test_results, test_targets)


    return 0


if __name__ == "__main__":
    main(sys.argv)
