from tensorflow.keras import layers, models
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
import itertools
from tqdm import tqdm
from datetime import datetime


def date_time():
    current_date_time = datetime.today().strftime("%Y%m%d_%H%M%S")
    return current_date_time


def build_model(filt_num1, filt_num2, num_of_dense, dense_len1, dense_len2, input_data, lr):  # model builder
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

    for i in range(num_of_dense):
        if i < 1:
            model.add(layers.Dense(units=dense_len1, use_bias=True))
            model.add(layers.PReLU())
        else:
            model.add(layers.Dense(units=dense_len2, use_bias=True))
            model.add(layers.PReLU())

    model.add(layers.Dense(1))
    model.add(layers.PReLU())

    model.compile(optimizer=tf.optimizers.Adam(learning_rate=lr),
                  loss=tf.keras.losses.MeanAbsolutePercentageError(name="MAPE"),
                  metrics=[tf.keras.metrics.MeanAbsoluteError(name="MAE")])
    return model


def models_prep(filt_num1, filt_num2, num_of_dense, dense_len1, dense_len2, input_data, lr):  # prepare all possible models
    combinations = np.array(list(itertools.product(filt_num1,
                                                   filt_num2,
                                                   num_of_dense,
                                                   dense_len1,
                                                   dense_len2)))
    modelz = list(map(lambda x: build_model(x[0], x[1], x[2], x[3], x[4], input_data, lr),
                      combinations))
    return modelz


def lr_schedule(epochs, lr):  # design learning rate strategy
    if epochs == 0:
        lr = 0.001  # RESET to the initial learning rate after the 100 epochs - IMPORTANT for cross validation
        lr = lr * 0.7  # decrease learning rate by 30%
        return lr
    else:
        if epochs % 10 != 0:
            lr = lr * 0.7  # decrease learning rate by 30% for 10 epochs - then RESET
        if epochs % 10 == 0:
            lr = (lr / 0.7 ** 9) * 0.8  # new price = starting price of last 10 epoch session - 20%
        return lr


def mini_report(model, rep_cols, results):  # result reporting method
    rep = pd.DataFrame(columns=rep_cols, index=range(len(results["loss"])))
    rep[rep_cols[0]] = model
    rep[rep_cols[1]] = range(len(results["loss"]))
    rep[rep_cols[2]] = results['loss']
    rep[rep_cols[3]] = results['val_loss']
    rep[rep_cols[4]] = results['MAE']
    rep[rep_cols[5]] = results['val_MAE']
    return rep


def grid_cv(model_partition, partion_num, tr_data, tr_label, folds, epochs, batch, callbacks_list, rep_cols):  # Grid search and Cross Validation
    now = date_time()
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
        report = report.append(mini_report(x + partion_num - 60, rep_cols, history.history), ignore_index=True)
    report.to_csv(f"report{partion_num}_{now}.csv")
    return report