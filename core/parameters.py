from typing import List
import numpy as np


class General_Params:

    def __init__(self,
                 window_length: int = 90,
                 given_epochs: int = 40,
                 given_initial_lr: float = 0.001,
                 given_minimum_lr: float = 0.00001,
                 given_reduction_rate: List = [0.99, 0.98, 0.95, 0.90],
                 reduction_freq_in_epochs: int = 20,
                 reset_rate_after_reduction: float = 0.7,
                 given_batch_size: int = 32,
                 given_folds: int = 10):

        self.slid_win_length = window_length
        self.epochs = given_epochs
        self.initial_lr = given_initial_lr
        self.minimum_lr = given_minimum_lr
        self.lr_reduction_rate = given_reduction_rate
        self.lr_reduction_frequency = reduction_freq_in_epochs
        self.lr_reset_rate = reset_rate_after_reduction
        self.batch_size = given_batch_size
        self.folds = given_folds

class Hyper_Params:

    def __init__(self,
                 number_of_filters_cov1: np.ndarray = np.arange(64, 257, 64),
                 number_of_filters_cov2: np.ndarray = np.arange(64, 257, 64),
                 extra_conv_layer_added: bool = [False, True],
                 number_of_filters_cov3: np.ndarray = np.arange(64, 257, 64),
                 number_of_units_dense1: np.ndarray = np.arange(64, 257, 64)):

        self.conv1_length = number_of_filters_cov1
        self.conv2_length = number_of_filters_cov2
        self.extra_conv_layer = extra_conv_layer_added
        self.conv3_length = number_of_filters_cov3
        self.dense1_length = number_of_units_dense1


