import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime

""" Imported files """
import parameters

""" Learning rate strategies """
class Learning_rate_strategy:
    def __init__(
            self, reduction_rate: float = 0.2, lowest_learning_rate: float = 0.0000017
            , epoch_to_reset: int = 20, learning_rate_reset_level: float = 0.7
    ):
        self.index = 0                                  # defines how many times LR has decreased
        self.reduc_rate = 1 - reduction_rate            # reduction rate of the learning rate
        self.floor_lr = lowest_learning_rate            # learning rate floor
        self.reset_epoch = epoch_to_reset               # reset every 20 epochs
        self.lr_reset_rate = learning_rate_reset_level  # after reset: go back to 70% of the last reset value

    """ CNN parameters """
    params = parameters.CNN_params()

    def lr_scheme(
            self, epoch, lr
    ):
        if epoch == 0:
            lr = self.params.initial_learning_rate  # very important step to RESET learning rate during the CV
            self.index = 0
            return lr

        elif lr > self.floor_lr and epoch % self.reset_epoch != 0:  # 2 conditions to reduce learning rate
            self.index += 1
            return lr * self.reduc_rate  # reduce LR by a specific percentage if the above conditions are True

        elif epoch % self.reset_epoch == 0:
            lr = (lr / self.reduc_rate ** (self.index - 1)) * self.lr_reset_rate  # RESET LR every a number of epochs
            self.index = 0
            return lr
        else:
            return lr  # waiting phase if LR is too low

    def lr_visual(
            self, show: bool = False
    ) -> None:
        now = datetime.today().strftime("%Y%m%d_%H%M%S")
        file = open(f"lr_logs_file_{now}.txt", "w")
        lr_only = []
        lr_sample = self.params.initial_learning_rate
        epochs = np.arange(0, self.params.epochs,  1)

        for epoch in epochs:
            if epoch == 0:
                lr_sample = self.params.initial_learning_rate
                lr_only.append(lr_sample)
                output = f"{lr_sample:06.8f}"
                file.write(f"Epoch = {epoch} -->  first epoch.   LR = {output}\n")
            elif lr_sample > self.floor_lr and epoch % self.reset_epoch != 0:
                lr_sample = lr_sample * self.reduc_rate
                lr_only.append(lr_sample)
                self.index += 1
                output = f"{lr_sample:06.8f}"
                file.write(f"Epoch = {epoch} --> reduction phase.   LR = {output}\n")
                file.write(f"reduction step = {self.index}\n")
            elif epoch % self.reset_epoch == 0:
                lr_sample = (lr_sample / self.reduc_rate ** (self.index - 1)) * self.lr_reset_rate
                lr_only.append(lr_sample)
                self.index = 0
                output = f"{lr_sample:06.8f}"
                file.write(f"Epoch = {epoch} --> reset phase.   LR = {output}\n")
            else:
                lr_sample = lr_sample
                lr_only.append(lr_sample)
                output = f"{lr_sample:06.8f}"
                file.write(f"Epoch = {epoch} --> waiting phase because LR is too low.   LR = {output}\n")
        file.close()

        plt.ioff()
        plt.close(fig=4)
        fig4, ax = plt.subplots(num=4, figsize=(13, 7), facecolor="#C7C5C5")
        ax.plot(lr_only, color="blue")
        ax.set_title("Learning rate strategy")
        plt.ylabel("Learning rate")
        plt.xlabel("Epochs")
        plt.savefig(f"LR_scheme_representation_{now}.png", facecolor=fig4.get_facecolor())
        if not show:
            pass
        else:
            fig4.show()


""" Test the LR strategy """
#
# # # # 1) LR callback   2) Callback to give LR after every epoch
# # # # comment out below to test one of the learning rate functions
# #
# EPOCHS = 100  # set the appropriate number of epochs
# learning_strategy = Learning_rate_strategy()

# # # comment out --> ONLY <-- the learning strategy you want to test
# test_lr_callback = tf.keras.callbacks.LearningRateScheduler(learning_strategy.lr_scheme)

# class CustomCallback(tf.keras.callbacks.Callback):
#     def on_epoch_end(self, epoch, logs=None):
#         print(f"{tf.keras.backend.eval(self.model.optimizer.lr):06.8f}")

# # # test model
# test_model = tf.keras.models.Sequential()
# test_model.add(tf.keras.layers.Dense(10))
# test_model.compile(tf.optimizers.Adam(learning_rate=0.0001), loss='mse',
#                    metrics=[tf.keras.metrics.MeanAbsoluteError(name="MAE")])
# sample_lr = f"{test_model.optimizer.lr.numpy():06.8f}"
#
# history = test_model.fit(np.arange(100).reshape(5, 20), np.zeros(5), epochs=EPOCHS,
#                          callbacks=[test_lr_callback, CustomCallback()], verbose=1)