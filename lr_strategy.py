import numpy as np
import tensorflow as tf


""" Learning rate strategy """
reduction_index = 0  # this is the global variable for the lr_scheme function


def lr_scheme(epoch, lr):  # the global variable reduction_index = 0 is NECESSARY
    global reduction_index
    if epoch == 0:
        lr = 0.0001  # very important step to RESET learning rate during the CV
        reduction_index = 0
        return lr

    elif lr > 0.00000017 and epoch % 20 != 0:  # 3 conditions to reduce learning rate
        reduction_index += 1
        return lr * 0.8  # reduce learning rate by 20% if the above conditions are ON

    elif epoch % 20 == 0:
        lr = (lr / 0.8 ** (reduction_index - 1)) * 0.7  # RESET LR every 20 epochs - go back to 70% of last peak
        reduction_index = 0
        return lr
    else:
        return lr  # waiting phase if LR is too low


""" Test for LR strategy """

# # 1) LR callback   2) Callback to give LR after every epoch
# # comment out below (up to the end) to test the learning rate function

# test_lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheme)
#
#
# class CustomCallback(tf.keras.callbacks.Callback):
#     def on_epoch_end(self, epoch, logs=None):
#         print(f"{tf.keras.backend.eval(self.model.optimizer.lr):06.8f}")
#
#
# # test model
# test_model = tf.keras.models.Sequential()
# test_model.add(tf.keras.layers.Dense(10))
# test_model.compile(tf.optimizers.Adam(learning_rate=0.0001), loss='mse')
# sample_lr = f"{test_model.optimizer.lr.numpy():06.8f}"
#
# history = test_model.fit(np.arange(100).reshape(5, 20), np.zeros(5), epochs=100,
#                          callbacks=[test_lr_callback, CustomCallback()], verbose=1)
