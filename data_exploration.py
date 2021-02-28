from typing import List, Tuple
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
import matplotlib.pyplot as plt
from math import ceil, floor
import logging
from sklearn import preprocessing

import data_preparation
import parameters


class Exploration(data_preparation.Utility_Functions):
    """
    This class is responsible for the data exploration phase and includes the following:
        1. Train - Test split
        2. Histograms (train and test)
        3. Descriptive statistics and Normality test
        4. Outlier detection
        5. Data normalisation
        6. Sliding Window application
    """

    def __init__(self, data: pd.DataFrame, running_mode: str = "full") -> None:
        self.input_data = data
        self.mode = running_mode
        self.window_length = parameters.General_Params().slid_win_length
        self.data = self.input_data.copy()

        if self.mode != "full":
            logging.warning("*** The FULL run is disabled: The data size and number of models is reduced ***")
            logging.warning("*** To be used only for functionality test ***")

    def split_train_test(self, split_rate: float = 0.7) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training set (70%) and test set (30%).

        When the running mode is NOT full:
        - The training set size is reduced to 5%.
        - The test set size is reduced to 2%.
        """
        if self.mode == "full":
            train_data = self.data[:ceil(len(self.data) * split_rate)]
            test_data = self.data[ceil(len(self.data) * split_rate):]
            return train_data, test_data
        else:
            train_data = self.data[:floor(len(self.data) * 0.05)]
            test_data = self.data[floor(len(self.data) * 0.05):floor(len(self.data) * 0.07)]
            return train_data, test_data

    @staticmethod
    def color_list() -> list:
        colors = ["black", "blue", "red", "green", "tan", "aqua",
                  "grey", "gold", "lime", "orchid", "orange", "brown"]
        return colors

    def distribution_comparison(self, *data: pd.Series, show: bool = True, save: bool = True) -> None:
        """
        Plot the probability distribution of the given data.
        Require at least on pd.Series for data.
        Can plot as many series as the color number in the color_list function.
        Optionally - Save the plot in a png file.
        """
        self.control_quantity(limit=len(self.color_list()), given_arguments=data)
        plt.ioff()
        plt.close(fig=3)
        now = datetime.today().strftime("%Y%m%d_%H%M%S")
        fig3 = plt.figure(num=3, figsize=(4, 3))
        for index in range(len(data)):
            data[index].plot(kind="kde", color=self.color_list()[index],
                             alpha=0.5, label=data[index].name, secondary_y=False)
        frame1 = plt.gca()
        frame1.axes.get_yaxis().set_visible(False)
        plt.legend()
        plt.ylabel("")
        if save:
            fig3.savefig(f"Distribution_Comparison_{now}.png")
        if show:
            fig3.show()

    def plot_series_but_ignore_date(self, *data: pd.Series, show: bool = True, save: bool = True) -> None:
        """
        Plot stock times-series without considering the price discontinuity because of the weekend days.
        Require at least on pd.Series for data.
        Can plot as many series as the color number in the color_list function.
        Optionally - Save the plot in a png file.
        """
        self.control_quantity(limit=len(self.color_list()), given_arguments=data)
        plt.ioff()
        plt.close(fig=4)
        now = datetime.today().strftime("%Y%m%d_%H%M%S")
        fig4 = plt.figure(num=4, figsize=(10, 4))
        ax0 = fig4.add_subplot(111)
        for index in range(len(data)):
            series_no_date = data[index].reset_index(drop=True)
            ax0.plot(series_no_date, color=self.color_list()[index], alpha=0.5, label=series_no_date.name)
        plt.tick_params(axis='x', bottom=False, top=False, labelbottom=False)
        plt.legend()
        if save:
            fig4.savefig(f"plot_on_demand_{now}.png")
        if show:
            fig4.show()

    @staticmethod
    def descriptive_stats_report(given_data: pd.DataFrame) -> pd.DataFrame:
        """ Create a report with descriptive stats of the data. """
        description = given_data.describe().T
        description.drop(columns=["count"], inplace=True)
        jarque_bera_test = given_data.apply(lambda attribute:
                                            stats.jarque_bera(attribute))
        description.insert(loc=1, column="median", value=given_data.median())
        description.insert(loc=2, column="jarque_bera", value=jarque_bera_test.iloc[0, 0:])
        description.insert(loc=3, column="p_value", value=jarque_bera_test.iloc[1, 0:])
        return description

    def custom_stat_report(self, data: pd.DataFrame, name: str, save: bool = True) -> pd.DataFrame:
        """
        Customise the index of the stat report.
        The name is added in the beginning of the index values.
        Optionally - Save the table in a xlsx file.
        """
        now = datetime.today().strftime("%Y%m%d_%H%M%S")
        description = self.descriptive_stats_report(given_data=data)
        description.index = list(map(lambda index_val: name + " " + index_val, description.index))
        if save:
            description.to_excel(f"test_stats{now}.xlsx")
        return description

    @staticmethod
    def box_plots(data: pd.DataFrame, show: bool = True, save: bool = True) -> None:
        """
        Design Box-Plots for outlier detection and data distribution representation.
        The fence constant is equal to 1.5.
        Require pd.Dataframe as data.
        Plot a box-plot for each data attribute.
        Optionally - Save the plot in a png file.
        """
        plt.ioff()
        plt.close(fig=5)
        now = datetime.today().strftime("%Y%m%d_%H%M%S")
        custom_marker = dict(alpha=0.1, markerfacecolor='r', marker='.')
        fig5 = plt.figure(num=5, figsize=(10, 4))
        for i in range(len(data.columns)):
            plt.rc('ytick', labelsize=6)
            ax0 = fig5.add_subplot(151+i)
            ax0.boxplot(data[data.columns[i]], flierprops=custom_marker)
            if data[data.columns[i]].max() > data[data.columns[i]].mean() * 4:
                ax0.set_ylim(0, 4 * data[data.columns[i]].mean())
            else:
                ax0.yaxis.set_ticks(np.linspace(data[data.columns[i]].min(),
                                                data[data.columns[i]].max(), 12))
            plt.tick_params(axis='x', bottom=False, top=False, labelbottom=False)
            plt.yticks(rotation=45)
            plt.xlabel(data.columns[i])
        fig5.tight_layout()
        if save:
            fig5.savefig(f"Box_Plots_{now}.png")
        if show:
            fig5.show()

    @staticmethod
    def scaler_min_max(data: pd.DataFrame, given_range: List = [0, 1]) -> np.ndarray:
        """
        Re-scale the data from 0 to 1.
        The range[0] is the minimum and the range[1] is the maximum.
        """
        method = preprocessing.MinMaxScaler(feature_range=(given_range[0], given_range[1]))
        scaled_data = method.fit_transform(data)
        return scaled_data

    def apply_sliding_window(self, scaled_data: np.ndarray, target_attribute: str = "Close") -> Tuple[list, list]:
        """
        Break data into consecutive segments.
        Returns:
            - list of segments.
            - list of target prices.
        Default window length = 90 (configuration is available in parameters.py (general parameters)).
        The transition np.ndarray -> pd.Dataframe provides better data manipulation.
        The default target prices arise from the "Close" attribute (modify by changing target_attribute).

        The last data segment is removed (used only for the last target price).
        The first target is removed (it is included in the initial data segment).
        """
        self.control_quantity(limit=self.window_length + 2, given_arguments=[scaled_data])
        self.control_type(expected_type=np.ndarray, given_arguments=[scaled_data])
        scaled_data = pd.DataFrame(data=scaled_data, columns=self.data.columns)
        sliding_data = list(map(lambda index:
                                scaled_data.iloc[index:index + self.window_length, 0:],
                                range(len(scaled_data) - (self.window_length - 1))))
        sliding_targets = list(map(lambda df:
                                   float(df[target_attribute].iloc[-1:]),
                                   sliding_data))
        sliding_data = sliding_data[:-1]
        sliding_targets = sliding_targets[1:]
        return sliding_data, sliding_targets

    def turn_dfs_into_arrays(self, given_data: np.ndarray,
                             given_target_attribute: str = "Close") -> Tuple[np.ndarray, np.ndarray]:
        sw_data, sw_targ = self.apply_sliding_window(scaled_data=given_data, target_attribute=given_target_attribute)
        sw_data = np.array(list(map(lambda df: np.array(df), sw_data)))
        sw_targ = np.array(sw_targ)
        return sw_data, sw_targ
