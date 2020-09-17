import pandas as pd
import numpy as np


def null_exist(df_frame):  # to test whether null values exist
    return df_frame.isna().all().all()


def type_consistency(df_frame):  # to test the data type consistency for each attribute
    df_types = df_frame.apply(lambda x:
                              x.apply(lambda y:
                                      type(y)).value_counts())
    return df_types.fillna(0)


def time_series_fillna(pd_frame):  # fill null values with the average of the last and next valid value
    pd_frame = (pd_frame.fillna(method="ffill") + pd_frame.fillna(method="bfill")) / 2
    return pd_frame


print("Data_preparation imported")