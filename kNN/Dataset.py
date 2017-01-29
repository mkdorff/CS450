import numpy as np


class Dataset(object):
    """ Skeletons of a dataset """

    # Is this ugly?
    def __init__(self, feature_names=np.array([]), target_names=np.array([]),
                 data=np.array([]), target=np.array([]), DESCR=""):
        self.feature_names = feature_names
        self.target_names = target_names
        self.data = data
        self.target = target
        self.DESCR = DESCR

    def set_feature_names(self, feature_names):
        self.feature_names = feature_names

    def set_target_names(self, target_names):
        self.target_names = target_names

    def set_data(self, data):
        self.data = data

    def set_target(self, target):
        self.target = target

    def set_DESCR(self, DESCR):
        self.DESCR = DESCR

    def load_from_txts_if_categorical(self, names_file, data_file):
        with open(names_file) as f:
            self.DESCR = f.readlines()

        # I transpose completely for my ease of thinking
        raw_data = np.genfromtxt(data_file, dtype=str, delimiter=',').T

        # Feature key as dict
        feature_key = {}
        feature_num = 0
        for nd_feature in raw_data:
            cat_value = 0
            feature = {}
            for nd_cat in np.unique(nd_feature):
                # cat = str(nd_cat)
                feature[str(nd_cat)] = cat_value
                cat_value += 1
            feature_key[feature_num] = feature
            feature_num += 1

        # does order matter? I don't see how we could change this dynamically
        self.feature_key = feature_key

        for y in range(len(raw_data)):
            for x in range(len(raw_data[y])):
                raw_data[y][x] = feature_key[y][str(raw_data[y][x])]

        self.data = raw_data[:len(raw_data) - 1].T.astype(np.int)
        self.target = raw_data[len(raw_data) - 1:].T.astype(np.int)

        return 0

    def load_from_txts_if_numerical(self, names_file, data_file):
        return 0

    def load_dataset_from_iris_csv(self, csv_file):
        csv = np.genfromtxt(csv_file, dtype=str, delimiter=",")
        for x in range(len(csv)):
            for y in range(len(csv[x])):
                csv[x][y] = csv[x][y].replace("\"", "")

        self.feature_names = csv[:1, 1:5]
        self.target_names = np.array(set(csv[1:, 5:6].flatten()))
        self.data = csv[1:, 1:5].astype(np.float)
        self.target = csv[1:, 5:6]

        for index in range(len(self.target)):
            if self.target[index] == 'setosa':
                self.target[index] = 0
            elif self.target[index] == 'versicolor':
                self.target[index] = 1
            elif self.target[index] == 'virginica':
                self.target[index] = 2
        self.target = self.target.flatten().astype(np.int)
