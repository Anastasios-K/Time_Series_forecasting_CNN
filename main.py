""" Imported libraries """
import pandas as pd
import tensorflow as tf
import numpy as np
import os
import random as rng

""" Imported files """
import prep_prep
import lr_strategy
import model_developement
import parameters
import model_refinement

np.random.seed(parameters.Randomness().seed_value)
rng.seed(parameters.Randomness().seed_value)
tf.random.set_seed(parameters.Randomness().seed_value)


df = pd.read_csv("GSK per min.csv")

""" Data cleaning and preparation """
preparation = prep_prep.Preparation(dataframe=df
                                    , active_columns=["Close", "Open", "High", "Low", "Volume"])
preparation.set_date_index()
preparation.sort_by_date()
preparation.drop_unused()
preparation.covert_to_float(comma_issue=True)
preparation.time_series_fillna()

""" Data preprocessing """
preprocess = prep_prep.Preprocessing(dataframe=preparation.df, running_mode="full")
train_df, test_df = preprocess.split_train_test()
# preprocess.original_moving_av(prices=train_df["Close"], date_values=pd.Series(train_df.index)) # DOULEUEI
# preprocess.distribution_comparison(train_prices=train_df["Close"], test_prices=test_df["Close"])
stats = preprocess.compare_descr(training_df=train_df, testing_df=test_df)

""" Whole sequense """
# Min_Max
slide_data_all_Mm, slide_label_all_Mm = preprocess.sliding_win_application(scaled_data
                                                                           =preprocess.scaler_min_max(train_df))
# z-score
slide_data_all, slide_label_all = preprocess.sliding_win_application(scaled_data
                                                                     =preprocess.scaler_z_score(train_df))
""" Daily analysis """
daily_train = preprocess.keep_whole_days(train_df)
# Min_Max
slide_data_daily_Mm, slide_label_daily_Mm = preprocess.sliding_win_application(scaled_data
                                                                             =preprocess.scaler_min_max(daily_train))
# z-score
slide_data_daily_Z, slide_label_daily_Z = preprocess.sliding_win_application(scaled_data
                                                                             =preprocess.scaler_z_score(daily_train))

""" Learning strategy """
checkpoint_path = "saved_models/cp.ckpt"

learning_strategy = lr_strategy.Learning_rate_strategy()
# learning_strategy.lr_visual(show=False)

callbacks = {"lr_callback": tf.keras.callbacks.LearningRateScheduler(learning_strategy.lr_scheme, verbose=0)
             , "early_stop_callback": tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
             , "save_model": tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path
                                                                , save_weights_only=True
                                                                , verbose=0
                                                                , save_best_only=True)}
""" Model development """
model_dev = model_developement.Model_dev(input_data=slide_data_all_Mm, running_mode=preprocess.mode)
combs = model_dev.all_combinations()
models = model_dev.build_all_models()

""" Training process -> Auto-Execution """
# model_dev.auto_grid_cv(training_data=sliding_data, targets=sliding_labels, callback_list=list(callbacks.values()))

""" Training process -> Manual-Execution 
    Comment out to run a specific partition only 
    Select and Set a valid partition from partitions dictionary before execution """
partitions = model_dev.dict_partitions()
model_dev.manual_grid_cv(partition=partitions["models60"], partition_num=1060, training_data=slide_data_all_Mm
                         , targets=slide_label_all_Mm, callback_list=list(callbacks.values()))
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





#
# new_hps = parameters.CNN_hyper_params(number_of_filters1=np.array([28]), number_of_filters2=np.array([20]), number_of_dense_layers=[2], length_dense1=np.array([48]), length_dense2=np.array([80]))
#
# model_dev_test = model_developement.Model_dev(input_data=slide_data_all_Mm, running_mode=preprocess.mode, model_hp_params=new_hps)
# # comb_test = model_dev_test.all_combinations()
# test_model = model_dev_test.build_all_models()
#
# model_dev.manual_grid_cv(partition=test_model, partition_num=1, training_data=slide_data_all_Mm, targets=slide_label_all_Mm, callback_list=list(callbacks.values()))
#
# test_model[0].save("my model")

# new_hps = parameters.CNN_hyper_params(number_of_filters1=np.array([28]), number_of_filters2=np.array([20]), number_of_dense_layers=[2], length_dense1=np.array([48]), length_dense2=np.array([80]))
# model_dev_test = model_developement.Model_dev(input_data=slide_data_all_Mm, running_mode=preprocess.mode, model_hp_params=new_hps)
# mod = model_dev_test.model_builder(28,20,2,48,80,1)
#
# from tensorflow.keras import layers, models
#
# mod = models.Sequential()
# mod.add(layers.Conv1D(filters=28
#                                 , kernel_size=5
#                                 , strides=1
#                                 , padding="same"
#                                 , use_bias=True
#                                 , input_shape=slide_data_all_Mm.shape[1:]
#                                 , kernel_initializer=tf.keras.initializers.GlorotNormal(seed=1)
#                                 , bias_initializer=tf.keras.initializers.GlorotNormal(seed=1)))
# mod.add(layers.PReLU(alpha_initializer=tf.keras.initializers.GlorotNormal(seed=1)))
# mod.add(layers.MaxPool1D())
#
# mod.add(layers.Conv1D(filters=20
#                                 , kernel_size=2
#                                 , strides=1
#                                 , padding="same"
#                                 , use_bias=True
#                                 , kernel_initializer=tf.keras.initializers.GlorotNormal(seed=1)
#                                 , bias_initializer=tf.keras.initializers.GlorotNormal(seed=1)))
# mod.add(layers.PReLU(alpha_initializer=tf.keras.initializers.GlorotNormal(seed=1)))
# mod.add(layers.MaxPool1D())
#
# mod.add(layers.Flatten())
#
# mod.add(layers.Dense(units=48
#                        , use_bias=True
#                        , kernel_initializer=tf.keras.initializers.GlorotNormal(seed=1)
#                        , bias_initializer=tf.keras.initializers.GlorotNormal(seed=1)))
# mod.add(layers.PReLU(alpha_initializer=tf.keras.initializers.GlorotNormal(seed=1)))
#
# mod.add(layers.Dense(units=80
#                        , use_bias=True
#                        , kernel_initializer=tf.keras.initializers.GlorotNormal(seed=1)
#                        , bias_initializer=tf.keras.initializers.GlorotNormal(seed=1)))
# mod.add(layers.PReLU(alpha_initializer=tf.keras.initializers.GlorotNormal(seed=1)))
#
# mod.add(layers.Dense(units=1
#                        , use_bias=True
#                        , kernel_initializer=tf.keras.initializers.GlorotNormal(seed=1)
#                        , bias_initializer=tf.keras.initializers.GlorotNormal(seed=1)))
# mod.add(layers.PReLU(alpha_initializer=tf.keras.initializers.GlorotNormal(seed=1)))
#
# mod.compile(optimizer=tf.optimizers.Adam(learning_rate=0.0001),
#                       loss=tf.keras.losses.MeanAbsolutePercentageError(name="MAPE"),
#                       metrics=[tf.keras.metrics.MeanAbsoluteError(name="MAE")
#                                , tf.keras.metrics.RootMeanSquaredError(name="RMSE")])
#
# a = mod.get_weights()
#
#
# model_dev.manual_grid_cv(partition=[mod], partition_num=1, training_data=slide_data_all_Mm, targets=slide_label_all_Mm, callback_list=list(callbacks.values()))
# model_dev.manual_grid_cv(partition=[best_mod], partition_num=1, training_data=slide_data_all_Mm, targets=slide_label_all_Mm, callback_list=list(callbacks.values()))
#





