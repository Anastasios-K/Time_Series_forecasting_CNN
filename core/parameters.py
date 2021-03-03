import yaml
import os
import numpy as np
from pathlib import Path

params_path = os.path.join(Path(__file__).parent.parent, "parameters_general.yaml")
hyper_params_path = os.path.join(Path(__file__).parent.parent, "parameters_hyper.yaml")


class General_Params:
    """ Read the paramameters_general.yaml and specify the general parameters of the entire project. """

    def __init__(self):
        with open(params_path) as file:
            params_dict = yaml.load(file, Loader=yaml.FullLoader)
            file.close()

        self.slid_win_length = params_dict["sliding_window_length"]
        self.epochs = params_dict["epochs"]
        self.initial_lr = params_dict["initial_learning_rate"]
        self.minimum_lr = params_dict["minimum_learning_rate"]
        self.lr_reduction_rate = params_dict["learning_rate_reduction_rate"]
        self.lr_reduction_frequency = params_dict["reduction_frequency_in_epochs"]
        self.lr_reset_rate = params_dict["reset_rate_after_reduction"]
        self.batch_size = params_dict["batch_size"]
        self.folds = params_dict["folds"]


class Hyper_Params:
    """ Read the paramameters_hyper.yaml and specify the hyper parameters of the CNN model. """

    def __init__(self):
        with open(hyper_params_path) as file:
            hyper_params_dict = yaml.load(file, Loader=yaml.FullLoader)
            file.close()

        self.conv1_length = np.array(hyper_params_dict["filters_convolution_layer_1"])
        self.conv2_length = np.array(hyper_params_dict["filters_convolution_layer_2"])
        self.extra_conv_layer = hyper_params_dict["third_convolution_layer_added"]
        self.conv3_length = np.array(hyper_params_dict["filters_convolution_layer_3"])
        self.dense1_length = np.array(hyper_params_dict["neurons_dense_layer_1"])
