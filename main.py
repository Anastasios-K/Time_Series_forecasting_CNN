import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"  # info and warning messages are not printed
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanAbsoluteError, MeanSquaredError

from core import data_preparation, data_exploration, learning_rate_strategy, model_developement

df = pd.read_csv(os.path.join(os.getcwd(), "data", "GSK per min.csv"))

""" Data Preparation """
prepare = data_preparation.Preparation(data=df)
prepare.set_timestamps_as_index()
prepare.sort_by_timestamp()
prepare.drop_unused()
prepare.transform_to_float()
prepare.time_series_fillna()
prepare.plot_daily_prices(show=True)
prepare.plot_prices_and_projection(show=True)

""" Data Exploration """
explore = data_exploration.Exploration(data=prepare.data)
train_set, test_set = explore.split_train_test()
explore.distribution_comparison(train_set["Close"], test_set["Close"], show=True)
explore.plot_series_but_ignore_date(train_set["Close"], test_set["Close"], show=True)
train_report = explore.custom_stat_report(data=train_set, name="Train")
test_report = explore.custom_stat_report(data=train_set, name="Test")
explore.box_plots(data=train_set, show=True)
scaled_train_data = explore.scaler_min_max(data=train_set)
scaled_test_data = explore.scaler_min_max(data=test_set)
sliding_window_train_data, sliding_window_train_target = explore.turn_dfs_into_arrays(given_data=scaled_train_data)
sliding_window_test_data, sliding_window_test_target = explore.turn_dfs_into_arrays(given_data=scaled_test_data)

""" Learning rate strategy """
learning_rate = learning_rate_strategy.Learning_Rate_Strategy()
learning_rate.plot_lr_strategy(show=True)
learning_rate.create_lr_log(strategy_index=1, save=True)
learning_rate_dict = learning_rate_strategy.create_lr_dict()

""" Model development """
for key in learning_rate_dict:
    model_dev = model_developement.Model_Development(input_data=sliding_window_train_data, folder_customisation=key)
    models_dict = model_dev.create_models_dict()
    model_dev.train_models(given_models=models_dict, training_targets=sliding_window_train_target,
                           lr_callback=learning_rate_dict[key])

""" Best Model Prediction """
model_selection = model_developement.Model_Selection()
best_model_char = model_selection.get_best_model_characteristics()
best_model = load_model(os.path.join(os.getcwd(),
                                     "saved_models" + best_model_char["best_lr_strategy"],
                                     "model_" + "1" + ".h5"))
prediction = best_model.predict([sliding_window_test_data])

""" Final evaluation """
mse = MeanSquaredError()
mae = MeanAbsoluteError()

rmse_eval = tf.sqrt(mse(sliding_window_test_target, prediction)).numpy()
mse_eval = mse(sliding_window_test_target, prediction).numpy()
mae_eval = mae(sliding_window_test_target, prediction).numpy()

print(f"RMSE = {rmse_eval}",
      f"MSE = {mse_eval}",
      f"MAE = {mae_eval}")

