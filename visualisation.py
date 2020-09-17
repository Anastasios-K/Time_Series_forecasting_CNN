import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def date_time():
    current_date_time = datetime.today().strftime("%Y%m%d_%H%M%S")
    return current_date_time


def closing_prices(closing_prices):  # plot closing prices
    now = date_time()
    plt.figure(1)
    figure = plt.plot(closing_prices)
    plt.savefig(f"Closing_prices_plot_{now}.png")



def original_prediction_represent(sl_data, sl_labels, sl_step):  # sl_step is a sample index from 0 to len(sl_data)
    now = date_time()
    plt.figure(2)
    figure = plt.plot(sl_data[sl_step][0])
    figure = plt.scatter(np.arange(0, len(sl_data[sl_step][0]), 1), sl_data[sl_step][0], c='black', s=8)
    figure = plt.scatter(len(sl_data[sl_step][0]) + 1, sl_labels[sl_step], c='r')
    plt.savefig(f"Original_Prediction_represent_{now}.png")



def lr_scheme_representation():
    plt.figure(3)
    now = date_time()
    epoch_sample = np.arange(0, 100, 1)
    LR = 0.0001
    REDUCTION_RATE = 0.8
    RESET_RATE = 0.7
    lr_only = []
    file = open(f"lr_logs_file_{now}.txt", "w")
    now = date_time()
    red_ind = 0

    for epoch in epoch_sample:
        if epoch == 0:
            lr_sample = LR
            lr_only.append(lr_sample)
            output = f"{lr_sample:06.8f}"
            file.write(f"Epoch = {epoch} -->  first epoch.   LR = {output}\n")
        elif lr_sample > 0.0000017 and epoch % 20 != 0:
            lr_sample = lr_sample * REDUCTION_RATE
            lr_only.append(lr_sample)
            red_ind += 1
            output = f"{lr_sample:06.8f}"
            file.write(f"Epoch = {epoch} --> reduction phase.   LR = {output}\n")
            file.write(f"reduction step = {red_ind}\n")
        elif epoch % 20 == 0:
            lr_sample = (lr_sample / REDUCTION_RATE ** (red_ind - 1)) * RESET_RATE
            lr_only.append(lr_sample)
            red_ind = 0
            output = f"{lr_sample:06.8f}"
            file.write(f"Epoch = {epoch} --> reset phase.   LR = {output}\n")
        else:
            lr_sample = lr_sample
            lr_only.append(lr_sample)
            output = f"{lr_sample:06.8f}"
            file.write(f"Epoch = {epoch} --> waiting phase because LR is too low.   LR = {output}\n")
    file.close()
    figure = plt.plot(lr_only)
    plt.savefig(f"LR_scheme_representation_{now}.png")



print("Visualisation imported")