import numpy as np


class kNN(object):
    """ A basic classifier that doesn't doesn't do too much """

    def __init__(self, training_data=None, training_targets=None):
        if training_data is not None and training_targets is not None:
            self.training_data = training_data
            self.training_targets = training_targets

    # That whole thing about choices. Not much training happens here huh?
    def train(self, training_data, training_targets):
        self.training_data = training_data
        self.training_targets = training_targets

        return 0

    # k is arbitrarily set here
    def predict(self, test_data, k=2):

        poopBucket = []
        for n in range(len(self.training_data)):
            meow = np.sum((self.training_data[n] - test_data[0]) ** 2)
            # print(meow)
            poopBucket.append(meow)

        print(np.array(poopBucket))
        print(np.argmin(np.array(poopBucket)))
        print(poopBucket[np.argmin(np.array(poopBucket))])
        # perfect, put k in there
        print(np.array(poopBucket).argsort()[:3])

        # The above works, I think. Now we gotta handle k

        # print(self.training_data[0])
        # print(test_data[0])
        # print(self.training_data[0] - test_data[0])
        # print((self.training_data[0] - test_data[0])**2)
        # print(np.sum((self.training_data[0] - test_data[0]) ** 2))


        # Are we emotionally prepared for 3 for loops?
        # for x in range(len(test_data)):
        #     distances = np.array([])
        #     for n in range(len(self.training_data)):

                # for x in range(len(test_data[y])):

        # distance of 0-3 elements in each row.




        # Makes sense to make an array for each item with the euclidean distance
        # standardized with z-scale...

        # I guess try to do the rb trees later if I had time, but I
        # didn't quite understand how that one worked. It seemed like there were holes
        test_results = [1 for x in range(len(test_data))]
        return np.array(test_results)
