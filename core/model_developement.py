from typing import List, Union
import os

os.environ['TF_DETERMINISTIC_OPS'] = "1"  # any bias, due to tf.nn.bias_add(), operates deterministically on GPU
os.environ['TF_CUDNN_DETERMINISTIC'] = "1"  # forces the selection of deterministic cuDNN convolution
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"  # info and warning messages are not printed

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanAbsolutePercentageError
from tensorflow.keras.metrics import MeanAbsoluteError, RootMeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.initializers import he_normal, zeros
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import pandas as pd
from math import ceil
import itertools
from tqdm import tqdm
import logging
import shutil

from core import parameters, data_preparation


class Training_Process_Functionality:
    """
    This class is to:
        A. create directories which are required in the training process.
        B. allow the training process to properly re-start after an unexpected interruption.
    In details, the class is responsible for:
        1. Making a directory to save model weights (customisation of dir name is optional).
        2. Making a directory to save model reports (customisation of dir name is optional).
        3. Keeping only the models and reports which are completed 100%.
        4. Removing all the uncompleted models and weights.
        5. Restarting the training process, skipping all models which have been already trained.
    """

    def __init__(self, folder_customisation: str = ""):
        self.dir_to_save_models = "saved_models" + folder_customisation
        self.dir_to_save_reports = "saved_reports" + folder_customisation

        if self.dir_to_save_models not in os.listdir():
            os.makedirs(f"{self.dir_to_save_models}")

        if self.dir_to_save_reports not in os.listdir():
            os.makedirs(f"{self.dir_to_save_reports}")

    @staticmethod
    def consider_model_id_only(element: str) -> None:
        return element[len("model_"):-len(".h5")]

    @staticmethod
    def consider_report_id_only(element: str) -> None:
        return element[len("report_"):-len(".csv")]

    @staticmethod
    def control_dir_to_check_parameter(given_param: str) -> None:
        if given_param not in ["models", "reports"]:
            raise ValueError("Invalid to_check value, should be models or reports")

    def collect_file_ids(self, dir_to_check: str) -> list:
        """
        Return the file IDs of the corresponding directory.
        The directory depends on the description of the dir_to_check parameter.
        dir_to_check should be models or reports, else a Value error is raised
        """
        self.control_dir_to_check_parameter(given_param=dir_to_check)
        if dir_to_check == "models" and self.dir_to_save_models in os.listdir():
            model_ids = list(map(lambda item:
                                 self.consider_model_id_only(element=item),
                                 os.listdir(self.dir_to_save_models)))
            return model_ids

        if dir_to_check == "reports" and self.dir_to_save_reports in os.listdir():
            report_ids = list(map(lambda item:
                                  self.consider_report_id_only(element=item),
                                  os.listdir(self.dir_to_save_reports)))
            return report_ids

    def find_common_ids(self) -> Union[list, None]:
        """
        Identify the common IDs between models and reports.
        """
        model_ids = self.collect_file_ids(dir_to_check="models")
        report_ids = self.collect_file_ids(dir_to_check="reports")
        if 0 in [len(model_ids), len(report_ids)]:
            common_ids = None
            return common_ids
        else:
            common_ids = list(filter(lambda item:
                                     item in report_ids,
                                     model_ids))
            return common_ids

    def keep_only_common_ids(self) -> None:
        """
        Keep only the common ids between models and reports.
            - remove the rest
        If the model or report folder is empty:
            - all models and reports are removed.
        """
        common_ids = self.find_common_ids()
        if not common_ids:
            # delete all (models & reports) if one of them is completely missing.
            list(map(lambda item:
                     os.remove(self.dir_to_save_models + "\\" + item),
                     os.listdir(self.dir_to_save_models)))
            list(map(lambda item:
                     os.remove(self.dir_to_save_reports + "\\" + item),
                     os.listdir(self.dir_to_save_reports)))
        else:
            # keep only the models and reports with common ids.
            list(map(lambda item:
                     os.remove(self.dir_to_save_models + "\\" + item)
                     if self.consider_model_id_only(element=item) not in common_ids
                     else None,
                     os.listdir(self.dir_to_save_models)))
            list(map(lambda item:
                     os.remove(self.dir_to_save_reports + "\\" + item)
                     if self.consider_report_id_only(element=item) not in common_ids
                     else None,
                     os.listdir(self.dir_to_save_reports)))

    def sort_given_dir_files(self, dir_to_check: str) -> Union[list, None]:
        """
        Sort the files of the corresponding directory, considering the file IDs only.
        The directory depends on the description of dir_to_check param.
        """
        self.control_dir_to_check_parameter(given_param=dir_to_check)
        if dir_to_check == "models":
            sorted_files = sorted(os.listdir(self.dir_to_save_models), reverse=False, key=self.consider_model_id_only)
            return sorted_files
        elif dir_to_check == "reports":
            sorted_files = sorted(os.listdir(self.dir_to_save_reports), reverse=False, key=self.consider_report_id_only)
            return sorted_files

    def delete_last_files(self) -> None:
        """
        Delete the last related file, before re-training execution.
        The last related file is deleted because it can be uncompleted, if the last training process interrupted.
        """
        sorted_models = self.sort_given_dir_files(dir_to_check="models")
        if sorted_models:
            os.remove(self.dir_to_save_models + "\\" + sorted_models[-1])
        sorted_reports = self.sort_given_dir_files(dir_to_check="reports")
        if sorted_reports:
            os.remove(self.dir_to_save_reports + "\\" + sorted_reports[-1])

    def control_training_models(self, all_models: dict) -> dict:
        """
        Apply all the above modifications.
        Determine the models which must be trained, considering the pre-existing models.
        1. The common IDs (models - reports) are kept only.
        2. The last saved file is removed to avoid uncompleted files.
        """
        self.keep_only_common_ids()
        self.delete_last_files()
        models_to_train = {}
        valid_existing_ids = self.collect_file_ids(dir_to_check="models")
        valid_numeric = np.array(list(map(lambda id_str:
                                          int(id_str),
                                          valid_existing_ids)))
        if len(valid_numeric) > 0:
            trainable_models_index = np.arange(start=valid_numeric.max() + 1, stop=len(all_models), step=1)
            list(map(lambda index:
                     models_to_train.update({list(all_models.keys())[index]: list(all_models.values())[index]}),
                     trainable_models_index))
        else:
            models_to_train = all_models
        return models_to_train


class Model_Development(parameters.Hyper_Params, parameters.General_Params,
                        data_preparation.Utility_Functions, Training_Process_Functionality):
    """
    This class is to:
        A. create all possible models - the models depend on the given hyper params.
        B. train only valid models - look at filter_valid_combinations method to find the validity requirements.
        C. save best weights of each model.
        D. save a report of the training progress for each model.
    """

    def __init__(self, input_data: np.ndarray, running_mode: str = "full", folder_customisation: str = ""):
        self.data = input_data
        self.mode = running_mode
        self.all_models = []
        parameters.Hyper_Params.__init__(self)
        parameters.General_Params.__init__(self)
        Training_Process_Functionality.__init__(self, folder_customisation=folder_customisation)

        if self.mode != "full":
            logging.warning("*** The FULL run is disabled: The data size and number of models is reduced ***")
            logging.warning("*** To be used only for functionality test ***")

    @staticmethod
    def generate_all_combinations() -> List[tuple]:
        """
        Calculate all possible combinations.
        The combinations depend on the given hyper params (parameters_hyper.yaml).
        Unspecified number of hyper params can be processed.
        The given params are extracted by parameters.py through the yaml file parameters_hyper.
        """
        combinations = list(itertools.product(*list(parameters.Hyper_Params().__dict__.values())))
        return combinations

    def turn_combinations_to_dict(self) -> List[dict]:
        """
        Create a dictionary of all the available combinations.
        The keys of the parameters.py are integrated with each available combination.
        """
        dict_keys = list(parameters.Hyper_Params().__dict__.keys())
        prep_empty_dict = list(map(lambda combo: {}, self.generate_all_combinations()))
        list(map(lambda dictionary, combo:
                 list(map(lambda index:
                          dictionary.update({dict_keys[index]: combo[index]}),
                          range(len(dict_keys)))),
                 prep_empty_dict, self.generate_all_combinations()))
        if self.mode == "partial":
            prep_empty_dict = prep_empty_dict[:10]
        return prep_empty_dict

    def filter_valid_combinations(self) -> List[dict]:
        """
        Keep only valid combinations.
        Valid combination requirements:
            1. the third convolution layer is active.
            2. the third convolution layer is disabled but its filter number takes the minimum available value.
            (the minimum value indicates that another parameter is currently changed in the model)
        """
        combinations = self.turn_combinations_to_dict()
        condition1 = list(filter(lambda comb:
                                 comb["extra_conv_layer"] is True,
                                 combinations))
        condition2 = list(filter(lambda comb:
                                 comb["extra_conv_layer"] is False
                                 and comb["conv3_length"] == self.conv1_length.min(),
                                 combinations))
        valid_combinations = condition1 + condition2
        return valid_combinations

    def model_builder(self, filter_size: int = 5, seed_val: int = 123, **kwargs) -> tf.keras.Sequential:
        """
        Build and compile a 1D-CNN depending on the given hyper params (parameters_hyper.yaml).
        Kwargs require a dict like below.
            {
                "conv1_length": int,
                "conv2_length": int,
                "extra_conv_layer": bool,
                "conv3_length": int,
                "dense1_length": int
            }
        """
        he_norm = he_normal(seed=seed_val)
        bias_val = zeros()

        model = models.Sequential()
        model.add(layers.Conv1D(filters=kwargs["conv1_length"],
                                kernel_size=filter_size,
                                strides=1,
                                padding="same",
                                use_bias=True,
                                input_shape=self.data.shape[1:],
                                kernel_initializer=he_norm,
                                bias_initializer=bias_val,
                                activation="relu"))
        model.add(layers.MaxPool1D())
        model.add(layers.Conv1D(filters=kwargs["conv2_length"],
                                kernel_size=ceil(filter_size / 2),
                                strides=1,
                                padding="same",
                                use_bias=True,
                                kernel_initializer=he_norm,
                                bias_initializer=bias_val,
                                activation="relu"))
        model.add(layers.MaxPool1D())
        if kwargs["extra_conv_layer"]:
            model.add(layers.Conv1D(filters=kwargs["conv3_length"],
                                    kernel_size=ceil(filter_size / 2),
                                    strides=1,
                                    padding="same",
                                    use_bias=True,
                                    kernel_initializer=he_norm,
                                    bias_initializer=bias_val,
                                    activation="relu"))
            model.add(layers.MaxPool1D())
        model.add(layers.Flatten())
        model.add(layers.Dense(units=kwargs["dense1_length"],
                               use_bias=True,
                               kernel_initializer=he_norm,
                               bias_initializer=bias_val,
                               activation="relu"))
        model.add(layers.Dense(units=1, use_bias=True,
                               kernel_initializer=he_norm,
                               bias_initializer=bias_val,
                               activation="relu"))
        model.compile(optimizer=Adam(learning_rate=parameters.General_Params().initial_lr),
                      loss=MeanAbsolutePercentageError(name="MAPE"),
                      metrics=[
                          MeanAbsoluteError(name="MAE"),
                          RootMeanSquaredError(name="RMSE")
                      ]
                      )
        return model

    def build_all_models(self) -> List[tf.keras.Sequential]:
        """ Build all models based on all possible combinations. """
        models = list(map(lambda combination_params:
                          self.model_builder(**combination_params),
                          tqdm(self.filter_valid_combinations())))
        self.all_models = models
        return self.all_models

    def create_models_dict(self, *given_models: list) -> dict:
        """ Create dictionary for given models. """
        model_dict = {}
        if not given_models:
            given_models = self.build_all_models()
        else:
            given_models = given_models[0]
        list(map(lambda index:
                 model_dict.update({str(index): given_models[index]}),
                 range(len(given_models))))
        return model_dict

    @staticmethod
    def create_report(model_id: int, results: dict) -> pd.DataFrame:
        """ Generate a report where the model training progress is saved. """
        report = pd.DataFrame(columns=["models", "epochs"])
        report["epochs"] = np.arange(start=1, stop=len(results["loss"]) + 1, step=1)
        report["models"] = model_id
        for key in results:
            report[key] = results[key]
        return report

    def train_models(self, given_models: dict, training_targets: np.ndarray, lr_callback):
        """
        Executes the whole training process.
        If the training process is interrupted:
            - create a new object and re-run the function (it detects automatically the already trained models).
        """
        self.control_type(expected_type=dict, given_arguments=[given_models])
        models = self.control_training_models(all_models=given_models)
        series_cross_val = list(TimeSeriesSplit(n_splits=parameters.General_Params().folds).split(self.data))
        for item in tqdm(models):
            model = models[item]
            for cross_val_element in series_cross_val:
                history = model.fit(x=self.data[:len(cross_val_element[0])],
                                    y=training_targets[:len(cross_val_element[0])],
                                    verbose=0,
                                    epochs=self.epochs,
                                    batch_size=self.batch_size,
                                    shuffle=False,
                                    validation_data=(self.data[:len(cross_val_element[1])],
                                                     training_targets[:len(cross_val_element[1])]),
                                    callbacks=[
                                        EarlyStopping(monitor='val_loss', patience=self.lr_reduction_frequency),
                                        ModelCheckpoint(
                                            filepath=os.getcwd() + f"\\{self.dir_to_save_models}\\model_{item}.h5",
                                            monitor="val_loss", verbose=0, save_best_only=True, mode="min"),
                                        lr_callback
                                    ]
                                    )
            report = self.create_report(model_id=item, results=history.history)
            report.to_csv(self.dir_to_save_reports + f"\\report_{item}.csv", index=False)


class Model_Selection:

    @staticmethod
    def multi_csv_to_dfs(given_dir: str) -> list:
        """ Convert given CSV files to dataframes. """
        dfs = list(map(lambda file:
                       pd.read_csv(os.path.join(given_dir, file)),
                       os.listdir(given_dir)))
        return dfs

    @staticmethod
    def create_final_dir():
        if "final_reports" in os.listdir():
            shutil.rmtree("final_reports")

        os.mkdir("final_reports")

    def create_final_reports(self) -> dict:
        final_rep_dict = {}
        self.create_final_dir()
        for item in os.listdir():
            if item.startswith("saved_reports"):
                learning_rate_id = item.split("saved_reports")[1]
                dfs = self.multi_csv_to_dfs(given_dir=item)
                if len(dfs) > 0:
                    final_df = pd.concat(dfs)
                    final_df.to_csv(os.path.join("final_reports", f"report_{learning_rate_id}.csv"), index=False)
                    final_rep_dict.update({"report_" + learning_rate_id: final_df})
        return final_rep_dict

    def get_best_model_characteristics(self) -> dict:
        best_char = {
            "best_performance": 2**32,
            "best_model": "",
            "best_lr_strategy": ""
        }
        final_reports_dict = self.create_final_reports()
        for item in final_reports_dict:
            performance = final_reports_dict[item]["val_loss"].min()
            if performance < best_char["best_performance"]:
                best_char["best_performance"] = performance
                best_char["best_model"] = final_reports_dict[item]["models"][
                    final_reports_dict[item]["val_loss"] == performance
                ]
                best_char["best_lr_strategy"] = item.split("report_")[1]
        return best_char

