import sys
import Dataset as ds

#
import numpy as np
import matplotlib.pyplot as plt

def main(argv):
    # Load & prep
    dataset = ds.Dataset()
    # dataset.load_iris()
    # dataset.load_lenses()
    # dataset.load_voting()
    dataset.load_chess()

    # plt.figure()
    # plt.plot([x + 1 for x in range(len(dataset.temp))], dataset.temp)
    # plt.plot([x + 1 for x in range(len(dataset.training_data[:, 0]))], dataset.training_data[:, 0])
    # plt.savefig("graphic.png")


    # Train, predict


    # How did we do?
    # dataset.report_accuracy()

    # Produce some text version of the decision tree here

    return 0


if __name__ == "__main__":
    main(sys.argv)
