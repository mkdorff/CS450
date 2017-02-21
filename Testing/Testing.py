import sys
import Dataset as ds
from sklearn.neural_network import MLPClassifier


def main(argv):
    # Load, randomize, set
    dataset = ds.Dataset()
    # dataset.load_from_txts('iris.names.txt', 'iris.data.txt')
    dataset.load_from_txts('pima-indians-diabetes.names.txt', 'pima-indians-diabetes.data.txt')
    dataset.randomize_data()
    dataset.split_data()
    dataset.standardize_data()

    X = dataset.training_data.tolist()
    y = dataset.training_targets.tolist()
    test = dataset.test_data.tolist()
    test_result = dataset.test_targets.tolist()


    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes = (3, 3), random_state = 1)
    clf.fit(X, y)

    predicted_results = clf.predict(test)

    correct = 0
    for i in range(len(test_result)):
        if test_result[i] == predicted_results[i]:
            correct += 1
    percentage = round(correct / len(test_result), 2) * 100
    print("Predicting targets at {}% accuracy".format(percentage))

    # print(["fizzbuzz" if (x+1) % 3 == 0 and (x+1) % 5 == 0 else "fizz" if (x+1) % 3 == 0 else "buzz" if (x+1) % 5 == 0 else (x+1) for x in range(100)])


    return 0


if __name__ == "__main__":
    main(sys.argv)
