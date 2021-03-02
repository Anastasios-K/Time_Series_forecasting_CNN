import unittest
import pandas as pd
import numpy as np
from datetime import datetime

from core import data_preparation


class Utility_Functions(unittest.TestCase):

    def test_control_quantity_for_valid_and_invalid_quantity(self):
        mock_object = data_preparation.Utility_Functions()
        mock_argument = pd.Series(data=np.ones(10))
        self.assertIsNone(mock_object.control_quantity(limit=1, given_arguments=[mock_argument]))
        with self.assertRaises(ValueError):
            mock_object.control_quantity(limit=1, given_arguments=[mock_argument, mock_argument])

    def test_control_argument_type_for_fully_and_mix_valid_and_invalid_types(self):
        mock_object = data_preparation.Utility_Functions()
        mock_series_type = pd.Series(data=np.ones(10))
        mock_array_type = np.array(np.ones(10))
        self.assertIsNone(mock_object.control_type(expected_type=pd.Series,
                                                   given_arguments=[mock_series_type]))
        with self.assertRaises(ValueError):
            mock_object.control_type(expected_type=list, given_arguments=[mock_series_type, mock_array_type])
        with self.assertRaises(ValueError):
            mock_object.control_type(expected_type=pd.Series, given_arguments=[mock_series_type,
                                                                               mock_array_type])


class Test_Preparation(unittest.TestCase):

    def test_check_attribute_existence_for_including_and_excluding_attributes(self):
        mock_df = pd.DataFrame(data=np.random.rand(10, 4), columns=["attr1", "attr2", "attr3", "attr4"])
        mock_object = data_preparation.Preparation(data=mock_df)
        with self.assertRaises(ValueError):
            mock_object.check_attribute_existence(attributes=["attr5", "attr6"])
        self.assertIsNone(mock_object.check_attribute_existence(attributes=["attr1", "attr2"]))
        self.assertIsNone(mock_object.check_attribute_existence(attributes=["attr3"]))

    def test_create_timestamps_for_given_date_and_time_in_string_format(self):
        mock_data = {
            "Local Date": pd.date_range(start="01-01-2021", end="20-01-2021",
                                        periods=10).to_series().dt.date.astype(str),
            "Local Time": pd.date_range(start="01-01-2021", end="20-01-2021",
                                        periods=10).to_series().dt.time.astype(str)
        }
        mock_df = pd.DataFrame(data=mock_data)
        expected = datetime.strptime("01-01-2021 00:00:00", "%d-%m-%Y %H:%M:%S")
        self.assertEqual(data_preparation.Preparation(data=mock_df).create_timestamps().iloc[0], expected)

    def test_create_timestamps_for_given_date_and_time_in_datetime_format(self):
        mock_data = {
            "Local Date": pd.date_range(start="01-01-2021", end="20-01-2021", periods=10).to_series().dt.date,
            "Local Time": pd.date_range(start="01-01-2021", end="20-01-2021", periods=10).to_series().dt.time
        }
        mock_df = pd.DataFrame(data=mock_data)
        expected = datetime.strptime("01-01-2021 00:00:00", "%d-%m-%Y %H:%M:%S")
        self.assertEqual(data_preparation.Preparation(data=mock_df).create_timestamps().iloc[0], expected)

    def test_set_timestamps_as_index_for_valid_dataframe_with_Date_and_Time_attributes(self):
        mock_data = {
            "Local Date": pd.date_range(start="01-01-2021", end="20-01-2021", periods=10).to_series().dt.date,
            "Local Time": pd.date_range(start="01-01-2021", end="20-01-2021", periods=10).to_series().dt.time
        }
        mock_df = pd.DataFrame(data=mock_data)
        expected = datetime.strptime("01-01-2021 00:00:00", "%d-%m-%Y %H:%M:%S")
        self.assertEqual(data_preparation.Preparation(data=mock_df).set_timestamps_as_index().index[0], expected)

    def test_drop_unused_for_dataframe_that_includes_attributes_out_of_scope(self):
        mock_df = pd.DataFrame(data=np.random.rand(10, 4), columns=["attr1", "Close", "attr2", "Open"])
        expected = ["Close", "Open"]
        self.assertListEqual(list(data_preparation.Preparation(data=mock_df).drop_unused().columns.values), expected)

    def test_sort_by_timestamp_for_given_index(self):
        mock_df = pd.DataFrame(data=np.zeros([3]), columns=["test_attribute"], index=[1010, 1127, 986])
        expected = 986
        self.assertEqual(data_preparation.Preparation(data=mock_df).sort_by_timestamp().index[0], expected)

    def test_transform_to_float_for_df_that_includes_string_with_comas_and_normal_float_values(self):
        mock_df = pd.DataFrame(data=[["3,0", 2.0], [3.7, "10,0.00"]], columns=["test_attribute1", "test_attribute2"])
        expected = [30.0, 3.7]
        outcome = data_preparation.Preparation(data=mock_df).transform_to_float()
        self.assertEqual(outcome["test_attribute1"].dtypes, "float")
        self.assertEqual(outcome["test_attribute1"].iloc[0], expected[0])
        self.assertEqual(outcome["test_attribute1"].iloc[1], expected[1])
        self.assertTrue(all(outcome.dtypes == float))

    def test_time_series_fillna_for_specific_filled_values(self):
        mock_df = pd.DataFrame(data=[5.0, np.nan, 2.0, np.nan, 8.0], columns=["test_attribute"])
        expected = [3.5, 5]
        outcome = data_preparation.Preparation(mock_df).time_series_fillna()
        self.assertEqual(outcome["test_attribute"].iloc[1], expected[0])
        self.assertEqual(outcome["test_attribute"].iloc[3], expected[1])

    def test_check_if_date_in_range_for_valid_and_invalid_date(self):
        date_index = pd.date_range(start="01-Jul-2020", end="15-Jul-2020", freq="H")
        mock_df = pd.DataFrame(data=np.random.rand(len(date_index)), columns=["attr1"], index=date_index)
        mock_object = data_preparation.Preparation(data=mock_df)
        self.assertIsNone(mock_object.check_if_date_in_range(given_prices=mock_df["attr1"], giv_year=2020, giv_month=7, giv_day=13))
        with self.assertRaises(ValueError):
            self.assertIsNone(mock_object.check_if_date_in_range(given_prices=mock_df["attr1"], giv_year=2019,
                                                                 giv_month=8, giv_day=23))

    def test_check_if_weekend_day_for_week_day_and_weekend_day(self):
        mock_object = data_preparation.Preparation(data=pd.DataFrame())
        week_day = dict(day=5, month=6, year=2020)  # it's Friday
        weekend_day = dict(day=6, month=6, year=2020)  # it's Saturday
        self.assertIsNone(mock_object.check_if_weekend_day(giv_day=week_day["day"], giv_month=week_day["month"],
                                                           giv_year=week_day["year"]))
        with self.assertRaises(ValueError):
            mock_object.check_if_weekend_day(giv_day=weekend_day["day"], giv_month=weekend_day["month"],
                                             giv_year=weekend_day["year"])

    def test_plot_daily_prices_for_valid_input(self):
        mock_object = data_preparation.Preparation(data=pd.DataFrame())
        date_index = pd.date_range(start="01-Jul-2020", end="15-Jul-2020", freq="H")
        test_series = pd.Series(data=np.random.rand(len(date_index)), index=date_index)
        in_range_date = dict(day=9, month=7, year=2020)
        self.assertIsNone(mock_object.plot_daily_prices(test_series, day=in_range_date["day"],
                                                        month=in_range_date["month"], year=in_range_date["year"]))

    def test_detect_projection_point_for_short_long_data(self):
        mock_object = data_preparation.Preparation(data=pd.DataFrame())
        short_series = pd.Series(data=np.random.rand(5))
        long_series = pd.Series(data=np.random.rand(200))
        self.assertEqual(mock_object.detect_projection_point(given_prices=long_series)[0], 194)
        self.assertEqual(mock_object.detect_projection_point(given_prices=long_series)[1], 195)
        with self.assertRaises(ValueError):
            mock_object.detect_projection_point(given_prices=short_series)

    def test_plot_prices_and_projection_for_valid_input(self):
        mock_object = data_preparation.Preparation(data=pd.DataFrame())
        date_index = pd.date_range(start="01-Jul-2020", end="15-Jul-2020", freq="H")
        mock_series = pd.Series(data=np.random.rand(len(date_index)), index=date_index)
        in_range_date = dict(day=9, month=7, year=2020)
        self.assertIsNone(mock_object.plot_prices_and_projection(mock_series, day=in_range_date["day"],
                                                                 month=in_range_date["month"],
                                                                 year=in_range_date["year"]))

if __name__ == "__main__":
    unittest.main()
