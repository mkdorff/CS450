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
