import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"  # info and warning messages are not printed

import unittest
import shutil
import os
import pandas as pd
import numpy as np
import random
import tensorflow as tf
import sys
sys.path.append("..")

from core import model_developement, parameters



# class Test_Training_Process_Functionality(unittest.TestCase):
#
#     def test_consider_model_id_only_for_valid_and_invalid_element(self):
#         mock_object = model_developement.Training_Process_Functionality()
#         valid_element = "model_101.h5"
#         invalid_element = "models_101.h5"
#         expected = "101"
#         self.assertEqual(mock_object.consider_model_id_only(element=valid_element), expected)
#         self.assertNotEqual(mock_object.consider_model_id_only(element=invalid_element), expected)
#         shutil.rmtree("saved_models")
#         shutil.rmtree("saved_reports")
#
#     def test_consider_report_id_only_for_valid_and_invalid_element(self):
#         mock_object = model_developement.Training_Process_Functionality()
#         valid_element = "report_101.csv"
#         invalid_element = "reports_101.csv"
#         expected = "101"
#         self.assertEqual(mock_object.consider_report_id_only(element=valid_element), expected)
#         self.assertNotEqual(mock_object.consider_report_id_only(element=invalid_element), expected)
#         shutil.rmtree("saved_models")
#         shutil.rmtree("saved_reports")
#
#     def test_control_dir_to_check_parameter_for_valid_and_invalid_param(self):
#         mock_object = model_developement.Training_Process_Functionality()
#         valid_element = ["models", "reports"]
#         invalid_element = "other"
#         self.assertIsNone(mock_object.control_dir_to_check_parameter(given_param=valid_element[0]))
#         self.assertIsNone(mock_object.control_dir_to_check_parameter(given_param=valid_element[1]))
#         with self.assertRaises(ValueError):
#             mock_object.control_dir_to_check_parameter(given_param=invalid_element)
#         shutil.rmtree("saved_models")
#         shutil.rmtree("saved_reports")
#
#     def test_collect_file_ids_for_valid_model_name(self):
#         mock_object = model_developement.Training_Process_Functionality(folder_customisation="_test_folder")
#         valid_file_name = 'model_101.h5'  # 6 chars before the ID and 3 chars after the ID
#         valid_file = open(os.path.join("saved_models_test_folder", valid_file_name), 'w')
#         valid_file.close()
#         expected = ["101"]
#         self.assertEqual(mock_object.collect_file_ids(dir_to_check="models"), expected)
#         shutil.rmtree("saved_models_test_folder")
#         shutil.rmtree("saved_reports_test_folder")
#
#     def test_collect_file_ids_for_invalid_model_name(self):
#         mock_object = model_developement.Training_Process_Functionality(folder_customisation="_test_folder")
#         invalid_file_name = 'the_model_101.h5'  # other than 6 chars before the IDa and 3 chars after the ID
#         invalid_file = open(os.path.join("saved_models_test_folder", invalid_file_name), 'w')
#         invalid_file.close()
#         expected = ["101"]
#         self.assertNotEqual(mock_object.collect_file_ids(dir_to_check="models"), expected)
#         shutil.rmtree("saved_models_test_folder")
#         shutil.rmtree("saved_reports_test_folder")
#
#     def test_collect_file_ids_for_valid_report_name(self):
#         mock_object = model_developement.Training_Process_Functionality(folder_customisation="_test_folder")
#         valid_file_name = 'report_101.cvs'  # 7 chars before the ID and 4 chars after the ID
#         valid_file = open(os.path.join("saved_reports_test_folder", valid_file_name), 'w')
#         valid_file.close()
#         expected = ["101"]
#         self.assertEqual(mock_object.collect_file_ids(dir_to_check="reports"), expected)
#         shutil.rmtree("saved_models_test_folder")
#         shutil.rmtree("saved_reports_test_folder")
#
#     def test_collect_file_ids_for_invalid_report_name(self):
#         mock_object = model_developement.Training_Process_Functionality(folder_customisation="_test_folder")
#         valid_file_name = 'the_report_101.cvs'  # 7 chars before the ID and 4 chars after the ID
#         valid_file = open(os.path.join("saved_reports_test_folder", valid_file_name), 'w')
#         valid_file.close()
#         expected = ["101"]
#         self.assertNotEqual(mock_object.collect_file_ids(dir_to_check="reports"), expected)
#         shutil.rmtree("saved_models_test_folder")
#         shutil.rmtree("saved_reports_test_folder")
#
#     def test_find_common_ids_for_3_models_and_3_files_but_only_2_common_IDs(self):
#         mock_object = model_developement.Training_Process_Functionality(folder_customisation="_test_folder")
#         for index in range(3):  # create files
#             temp_model_file_path = os.path.join("saved_models_test_folder", f"model_{index + 1}.h5")
#             temp_report_file_path = os.path.join("saved_reports_test_folder", f"report_{index}.csv")
#             with open(temp_model_file_path, "w+") as model_file:
#                 model_file.close()
#             with open(temp_report_file_path, "w+") as report_file:
#                 report_file.close()
#         expected = ["1", "2"]
#         self.assertListEqual(mock_object.find_common_ids(), expected)
#         shutil.rmtree("saved_models_test_folder")
#         shutil.rmtree("saved_reports_test_folder")
#
#     def test_keep_only_common_ids_for_3_models_and_3_files_but_only_2_common_IDs(self):
#         mock_object = model_developement.Training_Process_Functionality(folder_customisation="_test_folder")
#         for index in range(3):
#             temp_model_file_path = os.path.join("saved_models_test_folder", f"model_{index + 1}.h5")
#             temp_report_file_path = os.path.join("saved_reports_test_folder", f"report_{index}.csv")
#             with open(temp_model_file_path, "w+") as model_file:
#                 model_file.close()
#             with open(temp_report_file_path, "w+") as report_file:
#                 report_file.close()
#         expected_models = ["model_1.h5", "model_2.h5"]
#         expected_reports = ["report_1.csv", "report_2.csv"]
#         mock_object.keep_only_common_ids()
#         self.assertListEqual(os.listdir("saved_models_test_folder"), expected_models)
#         self.assertListEqual(os.listdir("saved_reports_test_folder"), expected_reports)
#         shutil.rmtree("saved_models_test_folder")
#         shutil.rmtree("saved_reports_test_folder")
#
#     def test_sort_given_dir_files_for_model_files(self):
#         mock_object = model_developement.Training_Process_Functionality(folder_customisation="_test_folder")
#         for index in range(3):
#             temp_model_file_path = os.path.join("saved_models_test_folder", f"model_{index}.h5")
#             with open(temp_model_file_path, "w+") as model_file:
#                 model_file.close()
#         expected_models = ["model_0.h5", "model_2.h5"]
#         sorted_directory = mock_object.sort_given_dir_files(dir_to_check="models")
#         self.assertEqual(sorted_directory[0], expected_models[0])
#         self.assertEqual(sorted_directory[-1], expected_models[-1])
#         shutil.rmtree("saved_models_test_folder")
#         shutil.rmtree("saved_reports_test_folder")
#
#     def test_delete_last_files_for_folders_of_2_models_and_2_reports(self):
#         mock_object = model_developement.Training_Process_Functionality(folder_customisation="_test_folder")
#         for index in range(3):
#             temp_model_file_path = os.path.join("saved_models_test_folder", f"model_{index}.h5")
#             report_file_path = os.path.join("saved_reports_test_folder", f"report_{index}.csv")
#             with open(temp_model_file_path, "w+") as model_file:
#                 model_file.close()
#             with open(report_file_path, "w+") as report_file:
#                 report_file.close()
#         expected_before = 3
#         expected_after = 2
#         self.assertEqual(len(os.listdir("saved_models_test_folder")), expected_before)
#         self.assertEqual(len(os.listdir("saved_reports_test_folder")), expected_before)
#         mock_object.delete_last_files()
#         self.assertEqual(len(os.listdir("saved_models_test_folder")), expected_after)
#         self.assertEqual(len(os.listdir("saved_reports_test_folder")), expected_after)
#         shutil.rmtree("saved_models_test_folder")
#         shutil.rmtree("saved_reports_test_folder")
#
#     def test_control_training_models_for_5_models_and_2_pretrained_models(self):
#         mock_object = model_developement.Training_Process_Functionality(folder_customisation="_test_folder")
#         mock_models_dict = {}
#         [mock_models_dict.update({f"{index}": np.random.rand(5)}) for index in range(5)]
#         for index in range(2):
#             temp_model_file_path = os.path.join("saved_models_test_folder", f"model_{index}.h5")
#             report_file_path = os.path.join("saved_reports_test_folder", f"report_{index}.csv")
#             with open(temp_model_file_path, "w+") as model_file:
#                 model_file.close()
#             with open(report_file_path, "w+") as report_file:
#                 report_file.close()
#         expected_model_quantity = 4
#         expected_starting_id = "1"
#         expected_final_id = "4"
#         mock_mod_control = mock_object.control_training_models(all_models=mock_models_dict)
#         self.assertEqual(len(mock_mod_control), expected_model_quantity)
#         self.assertEqual(list(mock_mod_control.keys())[0], expected_starting_id)
#         self.assertEqual(list(mock_mod_control.keys())[-1], expected_final_id)
#         shutil.rmtree("saved_models_test_folder")
#         shutil.rmtree("saved_reports_test_folder")
#
#
# class Test_Model_Developement(unittest.TestCase):
#
#     def test_generate_all_combinations_for_the_number_of_generated_combinations(self):
#         mock_object = model_developement.Model_Development(input_data=np.random.rand(10, 2))
#         mock_combinations = mock_object.generate_all_combinations()
#         expected = [len(element) for element in list(parameters.Hyper_Params().__dict__.values())]
#         expected = np.product(expected)
#         self.assertEqual(len(mock_combinations), expected)
#
#     def test_generate_all_combinations_for_the_first_and_last_value(self):
#         mock_object = model_developement.Model_Development(input_data=np.random.rand(10, 2))
#         mock_combinations = mock_object.generate_all_combinations()
#         expected_first = list(parameters.Hyper_Params().__dict__.values())[0][0]
#         expected_last = list(parameters.Hyper_Params().__dict__.values())[-1][-1]
#         self.assertEqual(mock_combinations[0][0], expected_first)
#         self.assertEqual(mock_combinations[-1][-1], expected_last)
#
#     def test_turn_combinations_to_dict_for_the_dict_keys_of_random_sample_from_the_generated_dicts(self):
#         mock_object = model_developement.Model_Development(input_data=np.random.rand(10, 2))
#         mock_comb_dict = mock_object.turn_combinations_to_dict()
#         expected = list(parameters.Hyper_Params().__dict__.keys())
#         self.assertListEqual(list(mock_comb_dict[random.randint(0, len(mock_comb_dict) - 1)].keys()), expected)
#         self.assertListEqual(list(mock_comb_dict[random.randint(0, len(mock_comb_dict) - 1)].keys()), expected)
#
#     def test_filter_valid_combinations_for_absence_of_invalid_combination_based_on_the_conditions(self):
#         mock_object = model_developement.Model_Development(input_data=np.random.rand(10, 2))
#         mock_comb_dict = mock_object.filter_valid_combinations()
#         random_dict = mock_comb_dict[random.randint(0, len(mock_comb_dict) - 1)]
#         self.assertFalse((random_dict["extra_conv_layer"] is False
#                           and random_dict["conv3_length"] >
#                           parameters.Hyper_Params().conv1_length.min()))
#
#     def test_model_builder_for_building_valid_model_depending_on_the_given_params(self):
#         mock_hyper_param_dict = {
#             "conv1_length": 64,
#             "conv2_length": 64,
#             "extra_conv_layer": True,
#             "conv3_length": 64,
#             "dense1_length": 64
#         }
#         mock_object = model_developement.Model_Development(input_data=np.random.rand(3, 100, 5))
#         mock_model = mock_object.model_builder(**mock_hyper_param_dict)
#         self.assertEqual(type(mock_model), tf.keras.Sequential)
#
#     def test_build_all_models_for_3_random_models_whose_type_should_be_kerasSequential(self):
#         mock_object = model_developement.Model_Development(input_data=np.random.rand(3, 100, 5), running_mode="partial")
#         mock_model_list = mock_object.build_all_models()
#         self.assertEqual(type(mock_model_list[2]), tf.keras.Sequential)
#         self.assertEqual(type(mock_model_list[3]), tf.keras.Sequential)
#
#     def test_create_models_dict_for_not_given_models(self):
#         mock_object = model_developement.Model_Development(input_data=np.random.rand(3, 100, 5), running_mode="partial")
#         expected = 4  # valid models are 4 instead of the 10 combinations in the partial mode
#         self.assertEqual(len(mock_object.create_models_dict()), expected)
#
#     def test_create_models_dict_for_2_given_models(self):
#         mock_object = model_developement.Model_Development(input_data=np.random.rand(3, 100, 5), running_mode="partial")
#         mock_models = ["model1", "model2"]
#         expected = 2
#         self.assertEqual(len(mock_object.create_models_dict(mock_models)), expected)
#
#     def test_create_report(self):
#         mock_object = model_developement.Model_Development(input_data=np.random.rand(3, 100, 5), running_mode="partial")
#         mock_results = {"loss": [10],
#                         "result2": [20]}
#         mock_report = mock_object.create_report(model_id=1, results=mock_results)
#         expected_attributes = ["models", "epochs", "loss", "result2"]
#         self.assertListEqual(list(mock_report.columns), expected_attributes)
#
class Test_Model_Selection(unittest.TestCase):
#
#     def test_multi_csv_to_dfs_for_2_csv_files(self):
#         mock_object = model_developement.Model_Selection()
#         mock_csvs_dir = "mock_csv_files"
#         os.mkdir(mock_csvs_dir)
#         mock_df1 = pd.DataFrame(data=np.array([[1, 2],
#                                                [3, 4]]), columns=["attr1", "attr2"])
#         mock_df2 = pd.DataFrame(data=np.array([[5, 6],
#                                                [7, 8]]), columns=["attr1", "attr2"])
#         mock_df1.to_csv(os.path.join(mock_csvs_dir, f"mock_csv1.csv"), index=False)
#         mock_df2.to_csv(os.path.join(mock_csvs_dir, f"mock_csv2.csv"), index=False)
#         expected = [list, pd.DataFrame]
#         self.assertEqual(type(mock_object.multi_csv_to_dfs(given_dir=mock_csvs_dir)), expected[0])
#         self.assertEqual(type(mock_object.multi_csv_to_dfs(given_dir=mock_csvs_dir)[0]), expected[1])
#         self.assertTrue(all(mock_object.multi_csv_to_dfs(given_dir=mock_csvs_dir)[0] == mock_df1))
#         shutil.rmtree("mock_csv_files")
#
#     def test_create_final_dir_when_dir_does_not_exist_in_advance(self):
#         mock_object = model_developement.Model_Selection()
#         mock_object.create_final_dir()
#         self.assertTrue("final_reports" in os.listdir())
#         shutil.rmtree("final_reports")
#
#     def test_create_final_dir_when_dir_exists_in_advance(self):
#         mock_object = model_developement.Model_Selection()
#         os.mkdir("final_reports")
#         mock_object.create_final_dir()
#         self.assertTrue("final_reports" in os.listdir())
#         shutil.rmtree("final_reports")
#
#     def test_create_final_reports_for_an_LR_strategy_with_id_78_and_2_saved_reports(self):
#         mock_object = model_developement.Model_Selection()
#         mock_dir_name = "saved_reports78"
#         os.mkdir(mock_dir_name)
#         mock_df1 = pd.DataFrame(data=np.array([[1, 2],
#                                                [3, 4]]), columns=["attr1", "attr2"])
#         mock_df2 = pd.DataFrame(data=np.array([[5, 6],
#                                                [7, 8]]), columns=["attr1", "attr2"])
#         mock_df1.to_csv(os.path.join(mock_dir_name, f"mock_csv1.csv"), index=False)
#         mock_df2.to_csv(os.path.join(mock_dir_name, f"mock_csv2.csv"), index=False)
#         expected = {"len": 4, "cols": 2, "value_1:0": 3, "value_3:1": 8}
#         mock_object.create_final_reports()
#         mock_final_rep_path = os.path.join("final_reports", "report_78.csv")
#         self.assertTrue("report_78.csv" in os.listdir("final_reports"))
#         self.assertEqual(len(pd.read_csv(mock_final_rep_path)), expected["len"])
#         self.assertEqual((len(pd.read_csv(mock_final_rep_path).columns)), expected["cols"])
#         self.assertEqual(pd.read_csv(mock_final_rep_path).iloc[1, 0], expected["value_1:0"])
#         self.assertEqual(pd.read_csv(mock_final_rep_path).iloc[3, 1], expected["value_3:1"])
#         self.assertEqual(type(mock_object.create_final_reports()), dict)
#         shutil.rmtree("saved_reports78")
#         shutil.rmtree("final_reports")

    def test_get_best_model_characteristics(self):
        print(os.getcwd())
        mock_object = model_developement.Model_Selection()
        mock_dirs = ["saved_reports78", "saved_reports98"]
        [os.mkdir(given_dir) for given_dir in mock_dirs]
        # create 3 mock reports, the 3rd on includes the best model
        mock_df1 = pd.DataFrame(columns=["models", "val_loss", "attr1"])
        mock_df1["models"] = np.arange(1, 11, 1)
        mock_df1["val_loss"] = np.ones(10)
        mock_df1["attr1"] = np.random.rand(10, 1)
        mock_df1.iloc[4, 1] = 0.5

        mock_df2 = pd.DataFrame(columns=["models", "val_loss", "attr1"])
        mock_df2["models"] = np.arange(1, 11, 1)
        mock_df2["val_loss"] = np.ones(10)
        mock_df2["attr1"] = np.random.rand(10, 1)
        mock_df2.iloc[2, 1] = 0.2

        mock_df3 = pd.DataFrame(columns=["models", "val_loss", "attr1"])
        mock_df3["models"] = np.arange(1, 11, 1)
        mock_df3["val_loss"] = np.ones(10)
        mock_df3["attr1"] = np.random.rand(10, 1)
        mock_df3.iloc[2, 1] = 0.1
        # use the mock reports to make 2 mock dirs; each one corresponds to one lr strategy
        mock_df1.to_csv(os.path.join(mock_dirs[0], f"mock_csv1.csv"), index=False)
        mock_df2.to_csv(os.path.join(mock_dirs[0], f"mock_csv2.csv"), index=False)

        mock_df1.to_csv(os.path.join(mock_dirs[1], f"mock_csv1.csv"), index=False)
        mock_df3.to_csv(os.path.join(mock_dirs[1], f"mock_csv2.csv"), index=False)

        expected ={"best_mod": 3, "best_perform": 0.1, "best_lr": "98"}
        self.assertEqual(mock_object.get_best_model_characteristics()["best_performance"], expected["best_perform"])
        self.assertEqual(mock_object.get_best_model_characteristics()["best_model"], expected["best_mod"])
        self.assertEqual(mock_object.get_best_model_characteristics()["best_lr_strategy"], expected["best_lr"])
        shutil.rmtree("saved_reports78")
        shutil.rmtree("saved_reports98")
        shutil.rmtree("final_reports")


if __name__ == "__main__":
    unittest.main()