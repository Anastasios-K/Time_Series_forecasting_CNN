from math import ceil
from tensorflow.keras import layers, models
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
import itertools
from tqdm import tqdm
from datetime import datetime

import model_development



def best_model_config(best_model):
    conv_filters = list(map(lambda z: z["config"]["filters"],
                            list(filter(lambda y: y["class_name"] == "Conv1D",
                                        list(map(lambda x: x, best_model.get_config()["layers"]))))))

    dense_filters = list(map(lambda z: z["config"]["units"],
                             list(filter(lambda y: y["class_name"] == "Dense",
                                         list(map(lambda x: x, best_model.get_config()["layers"]))))))

    dense_num = len(dense_filters) - 1

    return conv_filters, dense_filters, dense_num


def new_hyperparam(hp):
    new_hp = ceil(hp * 0.4)
    if new_hp % 2 == 0:
        pass
    else:
        new_hp += 1

    return new_hp


def build_refined_model(filt_num1, filt_num2, dense_len1, input_data, lr):  # model builder
    model = models.Sequential()
    model.add(layers.Conv1D(filters=filt_num1,
                            kernel_size=5,
                            strides=1,
                            padding="same",
                            use_bias=True,
                            input_shape=input_data.shape[1:]
                            ))
    model.add(layers.PReLU())
    model.add(layers.MaxPool1D())

    model.add(layers.Conv1D(filters=filt_num2,
                            kernel_size=2,
                            strides=1,
                            padding="same",
                            use_bias=True,
                            ))
    model.add(layers.PReLU())
    model.add(layers.MaxPool1D())

    model.add(layers.Flatten())

    model.add(layers.Dense(units=dense_len1, use_bias=True))
    model.add(layers.PReLU())

    model.add(layers.Dense(1))
    model.add(layers.PReLU())

    model.compile(optimizer=tf.optimizers.Adam(learning_rate=lr),
                  loss=tf.keras.losses.MeanAbsolutePercentageError(name="MAPE"),
                  metrics=[tf.keras.metrics.MeanAbsoluteError(name="MAE")])
    return model


def refined_models_prep(filt_num1, filt_num2, dense_len1, input_data, lr):  # prepare all possible models
    combinations = np.array(list(itertools.product(filt_num1,
                                                   filt_num2,
                                                   dense_len1)))
    refined_modelz = list(map(lambda x: build_refined_model(x[0], x[1], x[2], input_data, lr),
                      combinations))
    return refined_modelz



def refined_grid_cv(model_partition, partion_num, tr_data, tr_label, folds, epochs, batch, callbacks_list, rep_cols):  # Grid search and Cross Validation
    now = model_development.date_time()
    tscv = list(TimeSeriesSplit(n_splits=folds).split(tr_data))
    report = pd.DataFrame(columns=rep_cols)

    for x, model in enumerate(tqdm(model_partition)):
        for i in range(len(tscv)):
            history = model.fit(x=tr_data[:len(tscv[i][0])],
                                y=tr_label[:len(tscv[i][0])],
                                verbose=0,
                                epochs=epochs,
                                batch_size=batch,
                                shuffle=False,
                                validation_data=(tr_data[:len(tscv[i][1])], tr_label[:len(tscv[i][1])]),
                                callbacks=callbacks_list)
        report = report.append(model_development.mini_report(x + partion_num - 50, rep_cols, history.history), ignore_index=True)
    report.to_csv(f"ref_report{partion_num}_{now}.csv")
    return report
