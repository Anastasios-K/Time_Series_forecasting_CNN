import os
import pandas as pd
import numpy as np
from sklearn import preprocessing
from math import ceil
from datetime import datetime
import matplotlib.pyplot as plt
from scipy import stats
import scipy
import logging


class Preparation:

    def __init__(
            self, dataframe: pd.DataFrame, active_columns: callable(str)
    ) -> None:
        self.active_cols = active_columns
        self.df = dataframe

    def set_date_index(
            self
    ) -> pd.DataFrame:
        """ Creates time-stamp values """
        date_series = self.df["Local Date"].astype("str") + " " + self.df["Local Time"].astype("str")
        date_and_time = (date_series.apply(lambda x:
                                           datetime.strptime(x, "%d-%b-%Y %H:%M")))
        self.df.index = date_and_time
        return self.df

    def sort_by_date(
            self
    ) -> pd.DataFrame:
        """ Sort dataframe by date """
        self.df = self.df.sort_index(ascending=True)
        return self.df

    def drop_unused(
            self
    ) -> pd.DataFrame:
        """ Drop unused attributes """
        for col in self.df:
            if col not in self.active_cols:
                self.df = self.df.drop(columns=col)
        return self.df

    def covert_to_float(
            self, comma_issue: bool = True
    ) -> pd.DataFrame:
        """ Convert to float, if issue=True commas are replaced from every attribute """
        if not comma_issue:
            self.df = self.df.apply(lambda col: col.astype("float"))
            return self.df
        else:
            self.df = self.df.apply(lambda col:
                                    col.astype("str"))
            self.df = self.df.apply(lambda col:
                                    col.apply(lambda row:
                                              row.replace(",", "")))
            self.df = self.df.apply(lambda col:
                                    col.astype("float"))
            return self.df

    def time_series_fillna(
            self
    ) -> pd.DataFrame:
        """ Fill nan values with the average value of the last and next valid row
            Simplified fill-nan method for time series """
        self.df = (self.df.fillna(method="ffill") + self.df.fillna(method="bfill")) / 2
        return self.df


class Preprocessing:

    def __init__(
            self, dataframe: pd.DataFrame, running_mode: str = "full", window_length=180
    ) -> None:
        self.df = dataframe
        self.mode = running_mode
        if not (self.mode == "test" or self.mode == "full"):
            raise ValueError("Running mode is NOT valid. Please give \"test\" or \"full\"")
        self.win_len = window_length

    def split_train_test(
            self, split_rate: float = 0.7
    ) -> (pd.DataFrame, pd.DataFrame):
        """ Split data in train and test sets,
            test" mode -> returns only a short training data set (for functionality test only) """
        if self.mode == "full":
            train_data = self.df[:ceil(len(self.df) * split_rate)]
            test_data = self.df[ceil(len(self.df) * split_rate):].reset_index(drop=True)
            return train_data, test_data
        elif self.mode == "test":
            logging.warning("*** reduced data sets are generated due to the test running mode ***")
            train_data_for_func = self.df[:ceil(len(self.df) * 0.05)]
            test_data_for_func = self.df[ceil(len(self.df) * 0.05):ceil(len(self.df) * 0.07)].reset_index(drop=True)
            return train_data_for_func, test_data_for_func

    def original_moving_avg(
            self, prices: pd.Series, date_values: pd.Series, short_period: int = 120
            , long_period: int = 480, show: bool = False
    ) -> None:
        """ Plot original (close) prices combined with a short-term and a long-term moving average
            default values: short=120 (2 hours) - long=480 (whole trading day) """
        plt.ioff()
        plt.close(fig=1)  # resets the figure each time
        now = datetime.today().strftime("%Y%m%d_%H%M%S")
        ma_short = MA(np.array(prices), timeperiod=short_period, matype=0)
        ma_long = MA(np.array(prices), timeperiod=long_period, matype=0)
        fig1, ax = plt.subplots(num=1, figsize=(13, 7), facecolor="#C7C5C5")
        ax.plot(np.array(prices), alpha=0.3, color="green", label="original_prices")
        ax.plot(ma_short, color="blue", label="short moving average")
        ax.plot(ma_long, color="red", label="long moving average")
        plt.xticks(np.linspace(0, len(date_values), 3),
                   [date_values.iloc[0], date_values.iloc[ceil(len(date_values) / 2)],
                    date_values.iloc[-1]])
        leg_data = ax.get_legend_handles_labels()  # returns 2 tuples 0:list of graph lines - 1:list of label names
        ax.legend([leg_data[0][0], leg_data[0][1], leg_data[0][2]],
                  [leg_data[1][0], leg_data[1][1], leg_data[1][2]])
        ax.set_title("Short and Long moving average")
        plt.savefig(f"Short_Long_MA_and_Original_prices_{now}.png", facecolor=fig1.get_facecolor())
        if show:
            fig1.show()

    def distribution_comparison(
            self, train_prices: pd.Series
            , test_prices: pd.Series, show: bool = False
    ) -> None:
        """ Plot the distribution of two attributes (train, test) together
            Use it for comparison purpose """
        plt.ioff()
        plt.close(fig=2)
        now = datetime.today().strftime("%Y%m%d_%H%M%S")
        fig2 = plt.figure(num=2, figsize=(4, 3), facecolor="#C7C5C5")
        train_prices.plot(kind="kde", color="blue", label="training set")
        test_prices.plot(kind="kde", alpha=0.5, color="red", secondary_y=False, label="test set")
        plt.legend()
        plt.ylabel("")
        plt.title("Train - Test distribution comparison")
        plt.savefig(f"Distribution_Comparison_{now}.png")
        if show:
            fig2.show()

    def descriptive_stats(
            self, dataframe: pd.DataFrame
    ) -> pd.DataFrame:
        """
            Create a report with descriptive stats of the data
                                                               """
        description = dataframe.describe().T
        description.drop(columns=["count"], inplace=True)
        mode_val = dataframe.apply(lambda col: col.mode()).T.dropna(axis=1, how="any")
        jb_test = dataframe.apply(lambda col: stats.jarque_bera(col)[0])
        jb_test_pvals = dataframe.apply(lambda col: stats.jarque_bera(col)[1])
        description.insert(loc=1, column="mode", value=mode_val[mode_val.columns[0]].values)
        description.insert(loc=2, column="jarque_bera", value=jb_test)
        description.insert(loc=3, column="p_value", value=jb_test_pvals)
        return description

    def compare_descr(
            self, training_df: pd.DataFrame, testing_df: pd.DataFrame
    ) -> pd.DataFrame:
        """ Compare the descriptive stats of training and test sets """
        train_descr = self.descriptive_stats(training_df)
        test_descr = self.descriptive_stats(testing_df)
        train_descr.index = list(map(lambda col_name: "train " + col_name, train_descr.index))
        test_descr.index = list(map(lambda col_name: "test " + col_name, test_descr.index))
        final_df = pd.concat([train_descr,
                              pd.DataFrame(data=np.zeros([1, len(train_descr.columns)]), columns=train_descr.columns),
                              test_descr])
        return final_df

    def insert_date(
            self, dataframe: pd.DataFrame
    ) -> pd.DataFrame:
        """ Insert date attribute """
        data = dataframe.copy()
        data["Date"] = data.index.date
        return data

    def split_data_by_day(
            self, dataframe: pd.DataFrame
    ) -> list:
        """ Split time_series by day (date) """
        data = self.insert_date(dataframe)
        date_count = data["Date"].value_counts().sort_index()
        daily_dfs = list(map(lambda x:
                             data[data["Date"] == date_count.index[x]]
                             , range(len(date_count))))
        daily_dfs = list(map(lambda df:
                             df.drop(columns="Date")
                             , daily_dfs))
        return daily_dfs

    def keep_whole_days(
            self, dataframe: pd.DataFrame
    ) -> list:
        """ Keep only days that fit the sliding window length """
        daily_dataframes = self.split_data_by_day(dataframe)
        whole_day_dfs = list(filter(lambda x:
                                    len(x) > self.win_len - 1
                                    , daily_dataframes))
        return whole_day_dfs

    def scaler_min_max(
            self, data
    ) -> list:
        """ Min_Max rescaling from 0 to 1 """
        method = preprocessing.MinMaxScaler(feature_range=(0, 1))
        if not type(data) == list:
            scaled_data = method.fit_transform(data)
        else:
            scaled_data = list(map(lambda x:  method.fit_transform(x), data))
        return scaled_data

    def scaler_z_score(
            self, data
    ) -> list:
        """ Re-scale by day using z-score """
        if not type(data) == list:
            scaled_data = np.array(data.apply(lambda col:
                                              scipy.stats.zscore(col)))
        else:
            scaled_data = list(map(lambda df:
                                   np.array(df.apply(lambda col:
                                                     scipy.stats.zscore(col)))
                                   , data))
        return scaled_data

    def sliding_window_process(
            self, scaled_data
    ) -> (np.ndarray, np.ndarray):
        """ Restructures the data based on the Sliding Window method """
        sliding_data = np.array(list(map(lambda x:
                                         scaled_data[x:x + self.win_len].T,
                                         range(len(scaled_data) - self.win_len))))
        sliding_labels = np.array(list(map(lambda x:
                                           x[0:1, -1:],
                                           sliding_data)))
        sliding_data = sliding_data[:-1]
        sliding_labels = sliding_labels[1:]
        return sliding_data, sliding_labels

    def sliding_window_application(
            self, scaled_data
    ) -> (list, list):
        """ Apply sliding window method """
        if not type(scaled_data) == list:
            data = scaled_data
            sliding_data, sliding_labels = self.sliding_window_process(scaled_data=data)
            return sliding_data, sliding_labels
        else:
            sliding_data_list = []
            sliding_labels_list = []
            for partition in scaled_data:
                sliding_data, sliding_labels = self.sliding_window_process(scaled_data=partition)
                sliding_data_list.append(sliding_data)
                sliding_labels_list.append(sliding_labels)
            return sliding_data_list, sliding_labels_list

    def original_prediction_represent(
            self, sliding_data: np.ndarray, sliding_labels: np.ndarray
            , date_values: pd.Series, shift: int = 1, show: bool = False
    ) -> None:
        """ Plots a sample of the original data and the corresponding predicted value """
        # DOULEUEI - ALLA THELEI FIXARISMA STA XTICKS LOGO TWN ALLAGWN
        # GIA AUTO TA X-TICKS EINAI COMMENTED OUT
        plt.ioff()
        plt.close(fig=3)
        now = datetime.today().strftime("%Y%m%d_%H%M%S")
        fig3, ax = plt.subplots(num=3, figsize=(13, 7), facecolor="#C7C5C5")
        ax.plot(sliding_data[shift][0])
        ax.scatter(np.arange(0, len(sliding_data[shift][0]), 1), sliding_data[shift][0], c='black', s=8,
                   label="given data")
        ax.scatter(len(sliding_data[shift][0]) + 1, sliding_labels[shift], c='r', label="predicted value")
        # plt.xticks(np.linspace(0, len(sliding_data[shift][0]) + 1, 3),
        #            [date_values.iloc[shift],
        #             date_values.iloc[shift + ceil(len(sliding_data[shift][0]) / 2)],
        #             date_values.iloc[shift + len(sliding_data[shift][0]) - 1]])
        plt.yticks([])
        leg_data = ax.get_legend_handles_labels()
        ax.legend([leg_data[0][0], leg_data[0][1]],
                  [leg_data[1][0], leg_data[1][1]])
        ax.set_title("Original - Forecasting representation")
        plt.savefig(f"Original_Prediction_represent_{now}.png")
        if show:
            fig3.show()
