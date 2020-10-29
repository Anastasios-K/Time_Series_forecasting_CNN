""" Imported libraries """
import os
import tensorflow_environments
tensorflow_environments.set_environments()  # set tensorflow envs before importing the library
import pandas as pd
import tensorflow as tf

""" Imported files """
import data_preparation
import lr_strategy
import model_developement

df = pd.read_csv("GSK per min.csv")

""" Standardaise randoness """
standardisation = model_developement.Standardise_Randomness()
standardisation.set_seeds()
standardisation.set_threads()

""" Data cleaning and preparation """
preparation = data_preparation.Preparation(dataframe=df
                                           , active_columns=["Close", "Open", "High", "Low", "Volume"])
preparation.set_date_index()
preparation.sort_by_date()
preparation.drop_unused()
preparation.covert_to_float(comma_issue=True)
preparation.time_series_fillna()

""" Data preprocessing """
preprocess = data_preparation.Preprocessing(dataframe=preparation.df, running_mode="full")
train_df, test_df = preprocess.split_train_test()
# preprocess.original_moving_av(prices=train_df["Close"], date_values=pd.Series(train_df.index)) # DOULEUEI
# preprocess.distribution_comparison(train_prices=train_df["Close"], test_prices=test_df["Close"])
# stats = preprocess.compare_descr(training_df=train_df, testing_df=test_df)

""" Whole sequence """
# Min_Max
slide_data_all_Mm, slide_label_all_Mm = preprocess.sliding_window_application(scaled_data
                                                                              =preprocess.scaler_min_max(train_df))
# z-score
slide_data_all_Zsc, slide_label_all_Zsc = preprocess.sliding_window_application(scaled_data
                                                                                =preprocess.scaler_z_score(train_df))

# """ Daily analysis """
# daily_train = preprocess.keep_whole_days(train_df)
# # Min_Max
# slide_data_daily_Mm, slide_label_daily_Mm = preprocess.sliding_win_application(scaled_data
#                                                                              =preprocess.scaler_min_max(daily_train))
# # z-score
# slide_data_daily_Z, slide_label_daily_Z = preprocess.sliding_win_application(scaled_data
#                                                                              =preprocess.scaler_z_score(daily_train))

""" Learning strategy & Callbacks """
learning_strategy = lr_strategy.Learning_rate_strategy()
# learning_strategy.lr_visual(show=False)
callbacks = {"lr_callback": tf.keras.callbacks.LearningRateScheduler(learning_strategy.lr_scheme, verbose=0)
             , "early_stop_callback": tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)}

""" Model development """
model_dev = model_developement.Model_Development(input_data=slide_data_all_Mm, running_mode=preprocess.mode)
models = model_dev.build_all_models(initialisation=model_dev.GN_initializer())

""" Training process -> Auto-Execution """
model_dev.auto_grid_cv(training_data=slide_data_all_Mm, targets=slide_label_all_Mm
                       , callback_list=list(callbacks.values()))

""" Training process -> Manual-Execution """
""" Comment out to run a specific partition only 
    Select and Set a valid partition from partitions dictionary before execution """
# partitions = model_dev.dict_partitions()

# model_dev.manual_grid_cv(partition=partitions["models60"], partition_num=60, training_data=slide_data_all_Mm
#                          , targets=slide_label_all_Mm, callback_list=list(callbacks.values()))
# model_dev.manual_grid_cv(partition=partitions["models120"], partition_num=120, training_data=slide_data_all_Mm
#                          , targets=slide_label_all_Mm, callback_list=list(callbacks.values()))
# model_dev.manual_grid_cv(partition=partitions["models180"], partition_num=180, training_data=slide_data_all_Mm
#                          , targets=slide_label_all_Mm, callback_list=list(callbacks.values()))
# model_dev.manual_grid_cv(partition=partitions["models240"], partition_num=240, training_data=slide_data_all_Mm
#                          , targets=slide_label_all_Mm, callback_list=list(callbacks.values()))
# model_dev.manual_grid_cv(partition=partitions["models300"], partition_num=300, training_data=slide_data_all_Mm
#                          , targets=slide_label_all_Mm, callback_list=list(callbacks.values()))
# model_dev.manual_grid_cv(partition=partitions["models360"], partition_num=360, training_data=slide_data_all_Mm
#                          , targets=slide_label_all_Mm, callback_list=list(callbacks.values()))
# model_dev.manual_grid_cv(partition=partitions["models420"], partition_num=420, training_data=slide_data_all_Mm
#                          , targets=slide_label_all_Mm, callback_list=list(callbacks.values()))
# model_dev.manual_grid_cv(partition=partitions["models480"], partition_num=480, training_data=slide_data_all_Mm
#                          , targets=slide_label_all_Mm, callback_list=list(callbacks.values()))
# model_dev.manual_grid_cv(partition=partitions["models540"], partition_num=540, training_data=slide_data_all_Mm
#                          , targets=slide_label_all_Mm, callback_list=list(callbacks.values()))


# refinement = model_refinement.Refinement(models)
# f_report = refinement.final_report()
# best_mod, mod_index = refinement.choose_best_model()
# best_mod.summary()
# best_params = refinement.best_model_config()
# best_weights = models[150].get_weights()



