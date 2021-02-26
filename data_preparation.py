from typing import List, Tuple, Union, Type, Any
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt


class Utility_Functions:

    @staticmethod
    def control_quantity(limit: int, given_arguments: Union[list, tuple]) -> None:
        """ Control the number of given arguments. """
        if len(given_arguments) > limit:
            raise ValueError("Too many arguments are given, only 1 argument is expected")

    @staticmethod
    def control_type(expected_type: type, given_arguments: Union[tuple, list]) -> None:
        """
        Control the type of the given arguments.
        Single of multiple, given arguments must be passed into a list.
        """
        checker = [isinstance(element, expected_type) for element in given_arguments]
        invalid_types = [type(given_arguments[index]) for index in range(len(checker)) if checker[index] is False]
        if all(checker) is False:
            raise ValueError(f"{invalid_types} data type is given but {expected_type} is expected")


class Preparation(Utility_Functions):

    def __init__(self, data: pd.DataFrame, attributes: List = ["Close", "Open", "High", "Low", "Volume"]):
        self.input_data = data
        self.attributes_of_interest = attributes
        self.data = self.input_data.copy()

    def check_attribute_existence(self, attributes: list) -> None:
        """
        Check whether the given attributes are included in the data.
        Single and multiple attributes can be processed.
        Even a single attribute name must be given in a list.
        If an is missing:
            - A Value error is raised which indicates all missing attributes.
        """
        checker = [name in self.data.columns for name in attributes]
        missing_attributes = [attributes[ind] for ind in range(len(attributes)) if checker[ind] is False]
        if all(checker) is False:
            raise ValueError(f"The following attributes are missing {missing_attributes}")

    def create_timestamps(self) -> pd.Series:
        """
        Create timestamps.
        The Local Date and Local Time attributes are required.
        """
        self.check_attribute_existence(attributes=["Local Date", "Local Time"])
        date_string = self.data["Local Date"].astype(str)
        time_string = self.data["Local Time"].astype(str)
        date_time = date_string + " " + time_string
        timestamps = date_time.apply(pd.Timestamp)
        return timestamps

    def set_timestamps_as_index(self) -> pd.DataFrame:
        """ Set timestamps as index. """
        self.data.index = self.create_timestamps()
        return self.data

    def drop_unused(self) -> pd.DataFrame:
        """ Filter out the out of scope attributes. """
        attributes_to_drop = list(filter(lambda attribute:
                                         attribute not in self.attributes_of_interest,
                                         self.data))
        self.data = self.data.drop(columns=attributes_to_drop)
        return self.data

    def sort_by_timestamp(self) -> pd.DataFrame:
        """ Sort data by timestamp. """
        self.data = self.data.sort_index(ascending=True)
        return self.data

    def transform_to_float(self) -> pd.DataFrame:
        """
        Transform data to float.
        If commas exists in data, they are removed.
        """
        self.data = self.data.apply(lambda column:
                                    column.apply(lambda element:
                                                 float(str(element).replace(",", ""))))
        return self.data

    def time_series_fillna(self) -> pd.DataFrame:
        """ Fill nan values with the average value of the last and next valid data point. """
        self.data = (self.data.fillna(method="ffill") + self.data.fillna(method="bfill")) / 2
        return self.data

    @staticmethod
    def check_if_date_in_range(given_prices: pd.Series, giv_year: int, giv_month: int, giv_day: int) -> None:
        """
        Check whether the date is in the range of the given data.
        Given_prices should be pd.Series with datetime index.
        """
        timestamp = pd.Timestamp(year=giv_year, month=giv_month, day=giv_day).date()
        if str(timestamp) not in given_prices.index:
            raise ValueError("The given date is out of range - Please give a valid date")

    @staticmethod
    def check_if_weekend_day(giv_year: int, giv_month: int, giv_day: int) -> None:
        """ Check whether the date is a weekend day. """
        week_day = pd.Timestamp(year=giv_year, month=giv_month, day=giv_day).weekday()
        if week_day in [5, 6]:
            raise ValueError("This is a weekend date - Please give a valid date")

    def plot_daily_prices(self, *prices: pd.Series, day: int = 5, month: int = 6, year: int = 2020,
                          show: bool = False, save: bool = False) -> None:
        """
        Plot the minute-wise prices from the given data which correspond to the specified date.
        The prices must be pd.Series with timestamp index.
        Optionally - Save the plot in a png file.
        """
        if not prices:
            prices = self.data["Close"]
        else:
            self.control_quantity(limit=1, given_arguments=prices)
            prices = prices[0]
        self.control_type(expected_type=pd.Series, given_arguments=[prices])
        self.check_if_date_in_range(given_prices=prices, giv_day=day, giv_month=month, giv_year=year)
        self.check_if_weekend_day(giv_day=day, giv_month=month, giv_year=year)
        prices = prices[str(pd.Timestamp(day=day, month=month, year=year).date())]

        plt.style.use('seaborn-dark')
        plt.ioff()
        plt.close(fig=1)
        now = datetime.today().strftime("%Y%m%d_%H%M%S")
        fig1 = plt.figure(num=1, figsize=(10, 4))
        ax0 = fig1.add_subplot(111)
        ax0.plot(prices)
        ax0.set_xlim(prices.index.min(), prices.index.max())
        plt.gcf().autofmt_xdate()
        if save:
            fig1.savefig(f"Closing_prices_plot_{now}.png")
        if show:
            fig1.show()

    @staticmethod
    def detect_projection_point(given_prices: pd.Series) -> Tuple[int, int]:
        """
        Detect the limit point of the actual prices.
        Detect the projection point.
        """
        if len(given_prices) < 9:
            raise ValueError("Too short data is given, should include at least 3 data points")
        else:
            actual_limit_point = len(given_prices) - 6
            prediction_point = len(given_prices) - 5
        return actual_limit_point, prediction_point

    def plot_prices_and_projection(self, *prices: pd.Series, day: int = 5, month: int = 6, year: int = 2020,
                                   show: bool = False, save: bool = False) -> None:
        """
        Plot a sample of closing prices and a projection of the future prices.
        Require pd.Series for prices.
        Optionally - Save the plot in a png file.
        """
        if not prices:
            prices = self.data["Close"]
        else:
            self.control_quantity(limit=1, given_arguments=prices)
            prices = prices[0]
        self.control_type(expected_type=pd.Series, given_arguments=[prices])
        self.check_if_date_in_range(given_prices=prices, giv_day=day, giv_month=month, giv_year=year)
        self.check_if_weekend_day(giv_day=day, giv_month=month, giv_year=year)
        prices = prices[str(pd.Timestamp(day=day, month=month, year=year).date())]
        actual_point, prediction_point = self.detect_projection_point(given_prices=prices)

        plt.ioff()
        plt.close(fig=2)
        now = datetime.today().strftime("%Y%m%d_%H%M%S")
        fig2 = plt.figure(num=2, figsize=(10, 4))
        ax0 = fig2.add_subplot(111)
        ax0.plot_date(x=prices[:actual_point].index, y=prices[:actual_point].values,
                      ms=2.5, marker=".", color='black', linestyle="solid", linewidth=0.4)
        ax0.plot_date(x=prices[actual_point:prediction_point].index,
                      y=prices[actual_point:prediction_point].values,
                      ms=4.5, marker=".", color='r')
        ax0.set_xlim(prices.index[0], prices.index[prediction_point + 4])
        plt.gcf().autofmt_xdate()
        if save:
            fig2.savefig(f"Closing_prices_and_Projection_{now}.png")
        if show:
            fig2.show()
