import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import unittest
import tensorflow as tf
import numpy as np
import sys
sys.path.append("..")

from core import learning_rate_strategy, parameters

class Test_Lr_Strategy_Learning_Rate_Strategy(unittest.TestCase):

    def test_control_strategy_index_for_in_range_and_out_of_range_index(self):
        test_object = learning_rate_strategy.Learning_Rate_Strategy()
        in_range_index = len(parameters.General_Params().lr_reduction_rate) - 1
        out_of_range_index = len(parameters.General_Params().lr_reduction_rate) + 1
        self.assertIsNone(test_object.control_strategy_index(given_index=in_range_index))
        with self.assertRaises(ValueError):
            test_object.control_strategy_index(given_index=out_of_range_index)

    def test_prepare_lr_values_for_the_first_reduction_strategy(self):
        strategy_index = 0
        epochs = 3
        lr_params = parameters.General_Params()
        test_object = learning_rate_strategy.Learning_Rate_Strategy()
        expected = [lr_params.initial_lr,
                    lr_params.initial_lr * lr_params.lr_reduction_rate[strategy_index] ** epochs]
        outcome = test_object.prepare_lr_values(strategy_index=strategy_index)
        self.assertEqual(outcome[0], expected[0])
        self.assertAlmostEqual(outcome[epochs], expected[1])
        self.assertGreater(outcome[lr_params.lr_reduction_frequency], outcome[lr_params.lr_reduction_frequency - 1])

    def test_control_params_quantity_for_valid_and_invalid_param_number(self):
        test_object = learning_rate_strategy.Learning_Rate_Strategy()
        self.assertTrue(test_object.control_params_quantity(param1=[1]))  # 1 param
        self.assertFalse(test_object.control_params_quantity(param1=[1], param2=[2]))  # 2 params

    def test_control_param_names_for_valid_invalid_and_partially_valid_names(self):
        test_object = learning_rate_strategy.Learning_Rate_Strategy()
        self.assertTrue(test_object.control_params_name(param1=[1]))  # fully mistaken
        self.assertFalse(test_object.control_params_name(lr_values=[1], lr_names=[2]))  # fully correct
        self.assertTrue(test_object.control_params_name(lr_values=[1], lr_attr=[2]))  # partial mistaken

    def test_control_params_type_for_valid_invalid_and_partially_valid_types(self):
        test_object = learning_rate_strategy.Learning_Rate_Strategy()
        self.assertTrue(test_object.control_params_type(param1=1))  # fully mistaken
        self.assertFalse(test_object.control_params_type(lr_values=[1], lr_names=[2]))  # fully correct
        self.assertTrue(test_object.control_params_type(lr_values=[1], lr_names=2))  # partial mistaken

    def test_check_all_requirements_for_invalid_inputs_only(self):
        test_object = learning_rate_strategy.Learning_Rate_Strategy()
        partially_valid = test_object.check_all_requirements(param=1)
        self.assertEqual(partially_valid["params_quantity_violation"], True)
        self.assertEqual(partially_valid["params_name_violation"], True)
        self.assertEqual(partially_valid["params_type_violation"], True)

    def test_check_all_requirements_for_input_with_valid_quantity_and_name_but_invalid_type(self):
        test_object = learning_rate_strategy.Learning_Rate_Strategy()
        partially_valid = test_object.check_all_requirements(lr_values=1, lr_names=[1])
        self.assertEqual(partially_valid["params_quantity_violation"], False)
        self.assertEqual(partially_valid["params_name_violation"], False)
        self.assertEqual(partially_valid["params_type_violation"], True)

    def test_check_all_requirements_for_valid_inputs_only(self):
        test_object = learning_rate_strategy.Learning_Rate_Strategy()
        partially_valid = test_object.check_all_requirements(lr_values=[1], lr_names=[1])
        self.assertEqual(partially_valid["params_quantity_violation"], False)
        self.assertEqual(partially_valid["params_name_violation"], False)
        self.assertEqual(partially_valid["params_type_violation"], False)

    def test_determine_values_and_names_for_invalid_input_data_type_invalid_param_name_and_invalid_param_quantity(self):
        test_object = learning_rate_strategy.Learning_Rate_Strategy()
        with self.assertRaises(ValueError):
            test_object.determine_values_and_names(lr_values=1, lr_names=[1])
        with self.assertRaises(ValueError):
            test_object.determine_values_and_names(lrr_values=[1], lr_names=[1])
        with self.assertRaises(ValueError):
            test_object.determine_values_and_names(lr_values=[1])

    def test_plot_lr_strategy_for_no_given_values_and_names(self):
        test_object = learning_rate_strategy.Learning_Rate_Strategy()
        self.assertIsNone(test_object.plot_lr_strategy())

    def test_plot_lr_strategy_for_single_strategy_values_and_name(self):
        test_object = learning_rate_strategy.Learning_Rate_Strategy()
        self.assertIsNone(test_object.plot_lr_strategy(lr_values=[test_object.prepare_lr_values(strategy_index=0)],
                                                       lr_names=[0.99]))

    def test_plot_lr_strategy_for_multiple_strategies_values_and_names(self):
        test_object = learning_rate_strategy.Learning_Rate_Strategy()
        self.assertIsNone(test_object.plot_lr_strategy(lr_values=[test_object.prepare_lr_values(strategy_index=0),
                                                                  test_object.prepare_lr_values(strategy_index=1)],
                                                       lr_names=[0.9, 0.98]))

    def test_create_lr_log_for_valid_strategy_index(self):
        test_object = learning_rate_strategy.Learning_Rate_Strategy()
        self.assertIsNone(test_object.create_lr_log(strategy_index=1))

    def test_learning_rate_app_for_lr_on_epoch_end_of_a_mock_model(self):
        # prepare parameter object
        params = parameters.General_Params()
        # prepare custome callback to save learning rate after each epoch
        class Test_Callback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs):
                logs["learning rate values"] = tf.keras.backend.eval(self.model.optimizer.lr)
        # prepare a mock keras model
        test_model = tf.keras.models.Sequential()
        test_model.add(tf.keras.layers.Dense(10))
        test_model.compile(tf.optimizers.Adam(learning_rate=params.initial_lr), loss='mse',
                           metrics=[tf.keras.metrics.MeanAbsoluteError(name="MAE")])
        # prepare the learning rate callback
        test_object = learning_rate_strategy.Learning_Rate_Strategy(0)
        test_lr_callback = tf.keras.callbacks.LearningRateScheduler(test_object.learning_rate_app)
        # fit the mock model considering all callbacks
        test_history = test_model.fit(x=np.arange(50).reshape(10, 5), y=np.zeros(10), epochs=40, verbose=0,
                                      callbacks=[test_lr_callback, Test_Callback()])
        self.assertAlmostEqual(test_history.history["learning rate values"][0], params.initial_lr)
        self.assertAlmostEqual(test_history.history["learning rate values"][5],
                               params.initial_lr * params.lr_reduction_rate[0] ** 5)


if __name__ == "__main__":
    unittest.main()
