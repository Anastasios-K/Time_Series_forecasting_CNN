""" Imported libraries """
import pandas as pd
import numpy as np
from math import ceil
import tensorflow as tf

#MALAKIES

""" Imported files """
import data_preparation
import data_preprocessing
import model_development
import lr_strategy
import visualisation
import best_model_selection
# import model_refinement

""" Parameters """
WINDOW_LENGTH = 180
REPORT_COLS = ["models", "epochs", "Train MAPE", "Val MAPE", "Train MAE", "Val MAE"]
K_FOLDS = 10

""" CNN hyper params """
# FILT_NUM1 = np.arange(16, 81, 32)
# FILT_NUM2 = np.arange(16, 81, 32)
FILT_NUM1 = np.arange(12, 61, 16)
FILT_NUM2 = np.arange(20, 70, 16)
NUM_OF_DENSE = [1, 2]
DENSE_LEN1 = np.arange(16, 113, 32)
DENSE_LEN2 = np.arange(16, 113, 32)
LR = 0.001

""" CNN params """
EPOCHS = 100
FURTHER_EPOCHS = 150
BATCH_SIZE = 128


# swap between "test" and "official"
# to execute either a FUNCTIONALITY TEST run or an OFFICIAL RUN
MODE = "official"

if MODE == "test":
    print("CAUTION -- this is a functionality test run --")
    print("Change MODE into \"official\" to have a full run")

data = pd.read_csv("GSK per min.csv")
df = data.copy()
df.head()  # original head

""" Data cleaning and preparation """

null_values = data_preparation.null_exist(df)  # look for null values
df_types = data_preparation.type_consistency(df)  # look for data type consistency

df["Volume"] = np.array(list(map(lambda x: x.replace(",", ""),  # volume from str to float
                                 df["Volume"])))
df["Volume"] = df["Volume"].astype(df_types["Volume"].idxmin())
df.drop(["Adjusted Close", "%Chg", "Local Date", "Local Time"], axis=1, inplace=True)  # drop useless features

# comment out below to see the updated head
# print(df.head())

# comment out below to get data description
# data_description = df.describe().transpose()
closing_prices_plot = visualisation.closing_prices(df["Close"])  # plot all closing prices of the main data

#  split into training and test data
train_data = df[:ceil(len(df["Close"]) * 0.7)]
test_data = df[ceil(len(df["Close"]) * 0.8):]

""" Preprocessing """

# re-scale data from 0 to 1
scaled_train = data_preprocessing.scaler(train_data).T
# converting data and labels to Sliding Window form
slide_data, slide_labels = data_preprocessing.sliding_data(scaled_train, WINDOW_LENGTH)
# plot a sample of closing prices and the target price
original_prediction_plot = visualisation.original_prediction_represent(slide_data, slide_labels, 1234)

""" Model Development """

# Generate all the possible models based on the CNN hyper params
all_models = model_development.models_prep(FILT_NUM1, FILT_NUM2, NUM_OF_DENSE, DENSE_LEN1, DENSE_LEN2, slide_data, LR)

# Design the callbacks
lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_strategy.lr_scheme, verbose=0)  # learning rate
early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=13)  # early stopping

# plot a learning rate representation
lr_plot = visualisation.lr_scheme_representation()

# Break all models down into smaller partitions
if MODE == "test":
    functionality_test_models = all_models[0:2]
else:
    models60 = all_models[0:60]
    models120 = all_models[60:120]
    models180 = all_models[120:180]
    models240 = all_models[180:240]
    models300 = all_models[240:300]
    models360 = all_models[300:360]
    models420 = all_models[360:420]
    models480 = all_models[420:480]
    models540 = all_models[480:]

# Run and each part separately and save its report
if MODE == "test":
    run_func_test = model_development.grid_cv(functionality_test_models, 1, slide_data, slide_labels,
                                              K_FOLDS, EPOCHS, BATCH_SIZE,
                                              [lr_callback, early_stop_callback], REPORT_COLS)
else:
    run60 = model_development.grid_cv(models60, 60, slide_data, slide_labels,
                                      K_FOLDS, EPOCHS, BATCH_SIZE,
                                      [lr_callback, early_stop_callback], REPORT_COLS)
    
    run120 = model_development.grid_cv(models120, 120, slide_data, slide_labels,
                                       K_FOLDS, EPOCHS, BATCH_SIZE,
                                       [lr_callback, early_stop_callback], REPORT_COLS)
                                       
    run180 = model_development.grid_cv(models180, 180, slide_data, slide_labels,
                                       K_FOLDS, EPOCHS, BATCH_SIZE,
                                       [lr_callback, early_stop_callback], REPORT_COLS)
    
    run240 = model_development.grid_cv(models240, 240, slide_data, slide_labels,
                                       K_FOLDS, EPOCHS, BATCH_SIZE,
                                       [lr_callback, early_stop_callback], REPORT_COLS)
    
    run300 = model_development.grid_cv(models300, 300, slide_data, slide_labels,
                                       K_FOLDS, EPOCHS, BATCH_SIZE,
                                       [lr_callback, early_stop_callback], REPORT_COLS)
    
    run360 = model_development.grid_cv(models360, 360, slide_data, slide_labels,
                                       K_FOLDS, EPOCHS, BATCH_SIZE,
                                       [lr_callback, early_stop_callback], REPORT_COLS)
                                    
    run420 = model_development.grid_cv(models420, 420, slide_data, slide_labels,
                                       K_FOLDS, EPOCHS, BATCH_SIZE,
                                       [lr_callback, early_stop_callback], REPORT_COLS)
    
    run480 = model_development.grid_cv(models480, 480, slide_data, slide_labels,
                                       K_FOLDS, EPOCHS, BATCH_SIZE,
                                       [lr_callback, early_stop_callback], REPORT_COLS)
    
    run540 = model_development.grid_cv(models540, 540, slide_data, slide_labels,
                                       K_FOLDS, EPOCHS, BATCH_SIZE,
                                       [lr_callback, early_stop_callback], REPORT_COLS)


final_rep = best_model_selection.gen_final_report()  # generate the final report
best_model, model_index = best_model_selection.choose_best(final_rep, all_models)  # recognise the best model

best_model.summary()  # summary the best model

""" Model refinement """

bm_conv_filters, bm_dense_units, bm_dense_num = model_refinement.best_model_config(best_model)  # get best model configuration

# design refined hyper parameters
refined_FILT_NUM1 = np.arange(int(bm_conv_filters[0] - model_refinement.new_hyperparam(bm_conv_filters[0])),
                              int(bm_conv_filters[0] + model_refinement.new_hyperparam(bm_conv_filters[0]) + 1),
                              int(model_refinement.new_hyperparam(bm_conv_filters[0]) / 2))
refined_FILT_NUM2 = np.arange(int(bm_conv_filters[1] - model_refinement.new_hyperparam(bm_conv_filters[1])),
                              int(bm_conv_filters[1] + model_refinement.new_hyperparam(bm_conv_filters[1]) + 1),
                              int(model_refinement.new_hyperparam(bm_conv_filters[1]) / 2))
refined_DENSE1 = np.arange(int(bm_dense_units[0] - model_refinement.new_hyperparam(bm_dense_units[0])),
                           int(bm_dense_units[0] + model_refinement.new_hyperparam(bm_dense_units[0]) + 1),
                           int(model_refinement.new_hyperparam(bm_dense_units[0]) / 2))

# build all possible models based on the refined hyper parameters
all_refined_models = model_refinement.refined_models_prep(refined_FILT_NUM1, refined_FILT_NUM2, refined_DENSE1, slide_data, LR)

# break refined models in partitions
if MODE == "test":
    functionality_test_ref_mod = all_refined_models[:2]
else:
    ref_mods50 = all_refined_models[:50]
    ref_mods100 = all_refined_models[50:100]
    ref_mods150 = all_refined_models[100:]

if MODE == "test":
    ref_run_func_test = model_refinement.refined_grid_cv(functionality_test_ref_mod, 1, slide_data, slide_labels,
                                                         K_FOLDS, EPOCHS, BATCH_SIZE,
                                                         [lr_callback, early_stop_callback], REPORT_COLS)
else:
    ref_run50 = model_refinement.refined_grid_cv(ref_mods50, 50, slide_data, slide_labels,
                                                 K_FOLDS, EPOCHS, BATCH_SIZE,
                                                 [lr_callback, early_stop_callback], REPORT_COLS)
    ref_run100 = model_refinement.refined_grid_cv(ref_mods100, 100, slide_data, slide_labels,
                                                  K_FOLDS, EPOCHS, BATCH_SIZE,
                                                  [lr_callback, early_stop_callback], REPORT_COLS)
    ref_run150 = model_refinement.refined_grid_cv(ref_mods150, 150, slide_data, slide_labels,
                                                  K_FOLDS, EPOCHS, BATCH_SIZE,
                                                  [lr_callback, early_stop_callback], REPORT_COLS)


final_ref_rep = best_model_selection.gen_final_refined_report()
best_ref_model, ref_model_index = best_model_selection.choose_best(final_ref_rep, all_refined_models)



