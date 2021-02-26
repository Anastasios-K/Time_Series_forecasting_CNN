import unittest
import numpy as np
import pandas as pd

import data_exploration
import parameters


class Test_Exploration(unittest.TestCase):

    def test_split_train_test_for_full_and_partial_mode(self):
        test_df = pd.DataFrame(data=np.random.rand(100, 2), columns=["attr1", "attr2"])
        test_object_full = data_exploration.Exploration(data=test_df, running_mode="full")
        test_object_partial = data_exploration.Exploration(data=test_df, running_mode="partial")
        expected_full = [70, 30]
        expected_partial = [5, 2]
        self.assertEqual(len(test_object_full.split_train_test()[0]), expected_full[0])
        self.assertEqual(len(test_object_full.split_train_test()[1]), expected_full[1])
        self.assertEqual(len(test_object_partial.split_train_test()[0]), expected_partial[0])
        self.assertEqual(len(test_object_partial.split_train_test()[1]), expected_partial[1])

    def test_color_list_for_output_colors(self):
        test_list = data_exploration.Exploration(data=pd.DataFrame(data=np.random.rand(10),
                                                                   columns=["attr1"])).color_list()
        expected = ["red", "aqua"]
        self.assertEqual(test_list[2], expected[0])
        self.assertEqual(test_list[5], expected[1])

    def test_distribution_comparison_for_4_datasets(self):
        test_df = pd.DataFrame(data=np.random.rand(10, 4), columns=["attr1", "attr2", "attr3", "attr4"])
        test_object = data_exploration.Exploration(data=test_df)
        self.assertIsNone(test_object.distribution_comparison(test_df["attr1"], test_df["attr2"], test_df["attr3"],
                                                              test_df["attr4"]))

    def test_plot_series_but_ignore_date_for_2_datasets(self):
        date_times = pd.date_range(start="04-Jan-2021", end="10-Jan-2021", freq="T")
        test_df = pd.DataFrame(data=np.random.rand(len(date_times), 2), columns=["attr1", "attr2"], index=date_times)
        test_object = data_exploration.Exploration(data=test_df)
        self.assertIsNone(test_object.distribution_comparison(test_df["attr1"], test_df["attr2"]))

    def test_descriptive_stats_report_for_output_data_type(self):
        test_df = pd.DataFrame(data=np.random.rand(10, 2), columns=["attr1", "attr2"])
        test_object = data_exploration.Exploration(data=test_df)
        expected = pd.DataFrame
        self.assertEqual(type(test_object.descriptive_stats_report(given_data=test_df)), expected)

    def test_descriptive_stats_report_for_statistics_results(self):
        test_df = pd.DataFrame(data=np.random.rand(10, 2), columns=["attr1", "attr2"])
        test_object = data_exploration.Exploration(data=test_df)
        expected = test_df["attr1"].mean()
        self.assertEqual(test_object.descriptive_stats_report(given_data=test_df).iloc[0, 0], expected)

    def test_custom_stat_report_for_customised_index(self):
        test_df = pd.DataFrame(data=np.random.rand(10, 2), columns=["attr1", "attr2"])
        test_object = data_exploration.Exploration(data=test_df)
        expected = "the given name attr1"
        outcome = test_object.custom_stat_report(data=test_df, name="the given name")
        self.assertEqual(outcome.index[0], expected)

    def test_box_plots_for_dataframe_with_4_attributes(self):
        test_df = pd.DataFrame(data=np.random.rand(10, 4), columns=["attr1", "attr2", "attr3", "attr4"])
        test_object = data_exploration.Exploration(data=test_df)
        self.assertIsNone(test_object.box_plots(data=test_df))

    def test_scaler_min_max(self):
        test_df = pd.DataFrame(data=[10, 20, 30], columns=["attr1"])
        test_object = data_exploration.Exploration(data=test_df)
        expected = [1, 0.5, 0]
        self.assertAlmostEqual(test_object.scaler_min_max(data=test_df)[:, 0].max(), expected[0])
        self.assertTrue(expected[1] in test_object.scaler_min_max(data=test_df))
        self.assertAlmostEqual(test_object.scaler_min_max(data=test_df)[:, 0].min(), expected[2])

    def test_apply_sliding_window_for_type_outcome_length_and_links_data_target(self):
        data_length = 200
        window_length = parameters.General_Params().slid_win_length
        test_df = pd.DataFrame(data=np.random.rand(data_length, 3), columns=["attr1", "Close", "attr2"])
        test_object = data_exploration.Exploration(data=test_df)
        df_out_data, df_out_targets = test_object.apply_sliding_window(scaled_data=np.array(test_df))
        array_out_data, array_out_targets = test_object.apply_sliding_window(scaled_data=np.array(test_df))
        self.assertEqual(len(df_out_data[0]), window_length)
        self.assertEqual(len(df_out_data[-1]), window_length)
        self.assertEqual(df_out_data[1]["Close"].iloc[-1], df_out_targets[0])
        self.assertEqual(df_out_data[5]["Close"].iloc[-1], df_out_targets[4])
        self.assertTrue(all(df_out_data[0] == array_out_data[0]))


if __name__ == "__main__":
    unittest.main()
