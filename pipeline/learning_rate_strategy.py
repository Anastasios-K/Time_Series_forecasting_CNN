from typing import List
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from math import floor
from tensorflow.keras.callbacks import LearningRateScheduler

from pipeline import parameters


class Learning_Rate_Strategy:

    def __init__(self, *reduction_rate_index_for_callback_only: int):
        """
        The "reduction_index_for_callback_ONLY" is used only for the applicable function which takes 2 params only.
        """
        self.index = 0
        self.lr_params = parameters.General_Params()
        if not reduction_rate_index_for_callback_only:
            self.reduction_index_for_callback_ONLY = 0
        elif reduction_rate_index_for_callback_only[0] < len(parameters.General_Params().lr_reduction_rate):
            self.reduction_index_for_callback_ONLY = reduction_rate_index_for_callback_only[0]
        else:
            raise ValueError("The given index is out of range")

    @staticmethod
    def control_strategy_index(given_index: int) -> None:
        """
        Ensure that the given index is in the range of the available reduction rate strategies.
        """
        if given_index not in range(len(parameters.General_Params().lr_reduction_rate)):
            raise ValueError(f"The given index is out of range, should be in range "
                             f"{len(parameters.General_Params().lr_reduction_rate)}")

    def prepare_lr_values(self, strategy_index: int) -> List[float]:
        """
        Prepare the learning rate values.
        It is used only for learning rate visual representation.
        The related parameters are determined in the parameters package.
        The strategy index indicates the corresponding reduction rate value (pre-determined in parameters).
        """
        self.control_strategy_index(given_index=strategy_index)
        reduction_factor = self.lr_params.lr_reduction_rate[strategy_index]
        lr_sample = self.lr_params.initial_lr
        learning_rate_values = []

        for epoch in np.arange(0, self.lr_params.epochs,  1):
            if epoch == 0:
                lr_sample = self.lr_params.initial_lr
                learning_rate_values.append(lr_sample)
            elif lr_sample > self.lr_params.minimum_lr and epoch % self.lr_params.lr_reduction_frequency != 0:
                lr_sample = lr_sample * reduction_factor
                learning_rate_values.append(lr_sample)
                self.index = self.index + 1
            elif epoch % self.lr_params.lr_reduction_frequency == 0:
                lr_sample = (lr_sample / reduction_factor ** (floor(self.index * self.lr_params.lr_reset_rate)))
                learning_rate_values.append(lr_sample)
                self.index = 0
            else:
                learning_rate_values.append(lr_sample)
        self.index = 0
        return learning_rate_values

    @staticmethod
    def control_params_quantity(**given_params) -> bool:
        """
        Check whether a valid number of parameters is given.
        To be valid: No params or 2 params should be given.
        """
        outcome = len(given_params) not in [0, 2]
        return outcome

    @staticmethod
    def control_params_name(**given_params) -> bool:
        """
        Check whether valid parameter names are given.
        To be valid: "lr_values" and "lr_names" should be given.
        """
        outcome = any([element not in ["lr_values", "lr_names"] for element in given_params.keys()])
        return outcome

    @staticmethod
    def control_params_type(**given_params) -> bool:
        """
        Check whether valid data type given.
        To be valid: list is required.
        """
        outcome = any([type(element) != list for element in given_params.values()])
        return outcome

    def check_all_requirements(self, **given_params) -> dict:
        """
        Gather and check all the above requirements.

        To add another control function:
            - Prepare the corresponding function above.
            - Add it into the condition dictionary below.
            - Add the error reaction dictionary into the error_message function below.
        """
        condition_dict = dict(
            params_quantity_violation=self.control_params_quantity(**given_params),
            params_name_violation=self.control_params_name(**given_params),
            params_type_violation=self.control_params_type(**given_params)
        )
        return condition_dict

    @staticmethod
    def error_message() -> dict:
        message_dict = dict(
            params_quantity_violation="Wrong number of parameters is given, no params or 2 params should be given.",
            params_name_violation="Unknown parameters are given, should be \"lr_values\" or \"lr_names\".",
            params_type_violation="The given type is not valid, should be list."
        )
        return message_dict

    def determine_values_and_names(self, **given_params):
        """
        If 2 params are given and No requirement is violated, use given to specify the strategy_values and
        strategy_names.
        Else, if no params are given and no requirement is violated, activate all the available learning rate
        strategies.
        Else, check and raise the corresponding errors.
        """
        if len(given_params.keys()) == 2 and not any(list(self.check_all_requirements(**given_params).values())):
            strategy_values = given_params["lr_values"]
            strategy_names = given_params["lr_names"]
            return strategy_values, strategy_names
        elif len(given_params.keys()) == 0 and not any(list(self.check_all_requirements(**given_params).values())):
            strategy_values = list(map(lambda index:
                                       self.prepare_lr_values(strategy_index=index),
                                       range(len(parameters.General_Params().lr_reduction_rate))))
            strategy_names = list(map(lambda element:
                                      str(element),
                                      parameters.General_Params().lr_reduction_rate))
            return strategy_values, strategy_names
        else:
            for item in self.check_all_requirements(**given_params):
                if self.check_all_requirements(**given_params)[item]:
                    raise ValueError(self.error_message()[item])

    def plot_lr_strategy(self, show: bool = False, save: bool = False,
                         **strategy_value_and_name: [List[list], List[str]]) -> None:
        """
        Plot the learning rate values of the corresponding learning rate strategies.
        It can work with multiple value lists and names.
        If no strategy and name are specified, all the available strategies are plotted.
        The number of strategies depends on the reduction rate values which are set at the parameters.py.

        The strategy_value_and_name values are specified by the "determine_values_and_names" function.

        To specify a strategy manually:
        - Use the "prepare_lr_values" function to pass learning rate values into a parameter called "lr_values".
        - This function returns a list, but it must be nested in another list as it is explained by the annotation.
        - Use a string as a strategy name and pass it into a parameter called "lr_names".

        Optionally - Save the plot in a png file.
        """
        strategy_values, strategy_names = self.determine_values_and_names(**strategy_value_and_name)
        plt.ioff()
        plt.close(fig=6)
        now = datetime.today().strftime("%Y%m%d_%H%M%S")
        fig6 = plt.figure(num=6, figsize=(10, 4))
        for index in range(len(strategy_values)):
            plt.plot(strategy_values[index], label=f"strategy {strategy_names[index]}")
        plt.ylabel("Learning rate")
        plt.xlabel("Epochs")
        plt.legend()
        if save:
            fig6.savefig(f"LR_Strategies_{now}.png")
        if show:
            fig6.show()

    def create_lr_log(self, strategy_index: int, save: bool = False) -> None:
        """
        Create a learning rate log which includes:
            - The learning rate value for each epoch.
            - An explanation for the specific value.

        Optionally - Export the log to a text.doc
        """
        self.control_strategy_index(given_index=strategy_index)
        reduction_factor = self.lr_params.lr_reduction_rate[strategy_index]

        if save:
            now = datetime.today().strftime("%Y%m%d_%H%M%S")
            file = open(f"lr_reduction_{self.lr_params.lr_reduction_rate[strategy_index]}_logs_file_{now}.txt", "w")
            lr_sample = self.lr_params.initial_lr

            for epoch in np.arange(0, self.lr_params.epochs,  1):
                if epoch == 0:
                    lr_sample = self.lr_params.initial_lr
                    output = f"{lr_sample:06.8f}"
                    file.write(f"Epoch = {epoch} -->  first epoch.   LR = {output}\n")
                elif lr_sample > self.lr_params.minimum_lr and epoch % self.lr_params.lr_reduction_frequency != 0:
                    lr_sample = lr_sample * reduction_factor
                    self.index = self.index + 1
                    output = f"{lr_sample:06.8f}"
                    file.write(f"Epoch = {epoch} --> reduction phase.   LR = {output}\n")
                    file.write(f"reduction step = {self.index}\n")
                elif epoch % self.lr_params.lr_reduction_frequency == 0:
                    lr_sample = (lr_sample / reduction_factor ** (floor(self.index - self.lr_params.lr_reset_rate)))
                    self.index = 0
                    output = f"{lr_sample:06.8f}"
                    file.write(f"Epoch = {epoch} --> reset phase.   LR = {output}\n")
                else:
                    lr_sample = lr_sample
                    output = f"{lr_sample:06.8f}"
                    file.write(f"Epoch = {epoch} --> waiting phase because LR is too low.   LR = {output}\n")
            self.index = 0
            file.close()

    def learning_rate_app(self, epoch, lr):
        """
        To be used in Tensorflow callbacks to design the corresponding learning rate strategy.
        """
        reduction_factor = self.lr_params.lr_reduction_rate[self.reduction_index_for_callback_ONLY]
        if epoch == 0:
            lr = self.lr_params.initial_lr
            self.index = 0
            return lr
        elif lr > self.lr_params.minimum_lr and epoch % self.lr_params.lr_reduction_frequency != 0:
            self.index += 1
            return lr * reduction_factor
        elif epoch % self.lr_params.lr_reduction_frequency == 0:
            lr = (lr / reduction_factor ** floor(self.index * self.lr_params.lr_reset_rate))
            self.index = 0
            return lr
        else:
            return lr


def create_lr_dict():
    lr_dict ={}
    lr_callbacks = [LearningRateScheduler(Learning_Rate_Strategy(index).learning_rate_app)
                    for index in range(len(parameters.General_Params().lr_reduction_rate))]
    lr_names = [str(element) for element in parameters.General_Params().lr_reduction_rate]
    [lr_dict.update({lr_names[index]: lr_callbacks[index]}) for index in range(len(lr_names))]
    return lr_dict

