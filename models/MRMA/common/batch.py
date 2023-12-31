import numpy as np

from .dataset import DatasetManager
from .utils import set_seed


class BatchManager:
    def __init__(self, kind, seed=1):
        set_seed(seed)
        dataset_manager = DatasetManager(kind)
        self.train_data = dataset_manager.get_train_data()
        self.valid_data = dataset_manager.get_valid_data()
        self.test_data = dataset_manager.get_test_data()

        self.n_user = int(
            max(
                np.max(self.train_data[:, 0]),
                np.max(self.valid_data[:, 0]), np.max(self.test_data[:,
                                                                     0]))) + 1
        self.n_item = int(
            max(
                np.max(self.train_data[:, 1]),
                np.max(self.valid_data[:, 1]), np.max(self.test_data[:,
                                                                     1]))) + 1
        self.mu = np.mean(self.train_data[:, 2])
        self.std = np.std(self.train_data[:, 2])
