import numpy as np

class Randomness:
    def __init__(self
                 , seed_val: int = 1):
        self.seed_value = seed_val

class CNN_params:
    def __init__(self
                 , epochs_number: int = 100
                 , batch: int = 125
                 , initial_lr: float = 0.0001
                 , k_folds: int = 10):
        self.epochs = epochs_number
        self.batch_size = batch
        self.initial_learning_rate = initial_lr
        self.folds = k_folds

class CNN_hyper_params:
    def __init__(self
                 , number_of_filters1: np.ndarray = np.arange(12, 61, 16)
                 , number_of_filters2: np.ndarray = np.arange(20, 70, 16)
                 , number_of_dense_layers: list([int]) = [1, 2]
                 , length_dense1: np.ndarray = np.arange(16, 113, 32)
                 , length_dense2: np.ndarray = np.arange(16, 113, 32)):
        self.filter_num1 = number_of_filters1
        self.filter_num2 = number_of_filters2
        self.num_of_dense = number_of_dense_layers
        self.dense_len1 = length_dense1
        self.dense_len2 = length_dense2
