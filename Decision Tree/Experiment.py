import sys
import Dataset as ds


def main(argv):
    # Load, randomize, split, [discretize]
    dataset = ds.Dataset()
    dataset.load_iris()
    print(dataset.data)
    dataset.randomize_data()
    dataset.split_data()

    # Train, predict


    # How did we do?
    dataset.report_accuracy()

    # Produce some text version of the decision tree here

    return 0


if __name__ == "__main__":
    main(sys.argv)
