import numpy as np


class kNN(object):
    """ A basic classifier that doesn't doesn't do too much """

    # Let's explicitly 'train' our classifier
    def train(self, training_data, training_targets):
        print("Standardizing & training on {} items".format(len(training_data)))

        # we're going to standardize our data here
        standardized_data = training_data.T
        col_means = []
        col_stds = []

        # that would be really embarrassing it this math was wrong
        for x in range(len(standardized_data)):
            col_means.append(np.mean(standardized_data[x]))
            col_stds.append(np.std(standardized_data[x]))
            standardized_data[x] = [(el - col_means[x]) / col_stds[x] for el in standardized_data[x]]

        self.training_data = standardized_data.T
        self.training_targets = training_targets
        self.means = np.array(col_means)
        self.stds = np.array(col_stds)

    # k is arbitrarily set here
    def predict(self, test_data, k=2):
        # Setup
        # standardize test data based off of training means and stdevs
        standardized_test_data = np.array([(x - self.means) / self.stds for x in test_data])
        test_results = []

        for test_item in standardized_test_data:
            NN_sums = []
            for training_row in self.training_data:
                sum = np.sum((training_row - test_item) ** 2)
                NN_sums.append(sum)

            NN_results = self.training_targets[np.array(NN_sums).argsort()[:k]]
            # apparently only works with ints, but I'll format results to ints
            items, freq = np.unique(NN_results, return_counts=True)
            NN_freq = np.asarray((items, freq)) # Tranpose?

            # simplest case first
            if len(NN_freq[0]) == 1:
                test_results.append(NN_freq[0][0])
            else:
                # this will stay a 2D array, man this is ugly
                sort = NN_freq[1].argsort()[-2:]
                NN_freq[0] = NN_freq[0][sort]
                NN_freq[1] = NN_freq[1][sort]

                mxidx = len(NN_freq) - 1
                if NN_freq[1][mxidx] > NN_freq[1][mxidx - 1]:
                    test_results.append(NN_freq[0][mxidx])
                else:
                    for x in NN_results:
                        if NN_results[x] == NN_freq[0][mxidx] or NN_results[x] == NN_freq[0][mxidx - 1]:
                            test_results.append(NN_results[x])
                            break
                    else:
                        # We should never reach here, 0 is completely arbitrary
                        test_results.append(0)

        print(np.array(test_results))
        return np.array(test_results)
