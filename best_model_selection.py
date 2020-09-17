import pandas as pd
import os
from datetime import datetime

def date_time():
    current_date_time = datetime.today().strftime("%Y%m%d_%H%M%S")
    return current_date_time


def gen_final_report():
    now = date_time()
    current_dir_list = os.listdir()
    reports = list(map(lambda y: pd.read_csv(y),
                       list(filter(lambda x: x[:6] == "report", current_dir_list))))
    final_report = pd.concat(reports, ignore_index=True)
    final_report = final_report.sort_values(by=["models", "epochs"])
    final_report.to_csv(f"final_report_{now}.csv")
    return final_report


def gen_final_refined_report():
    now = date_time()
    current_dir_list = os.listdir()
    reports = list(map(lambda y: pd.read_csv(y),
                       list(filter(lambda x: x[:10] == "ref_report", current_dir_list))))
    final_report = pd.concat(reports, ignore_index=True)
    final_report = final_report.sort_values(by=["models", "epochs"])
    final_report.to_csv(f"final_ref_report_{now}.csv")
    return final_report


def choose_best(report, models):
    best_mod_index = report["models"][
        int(report[report["Val MAPE"] == report["Val MAPE"].min()].index.values)
    ]
    best_model = models[best_mod_index]
    return best_model, best_mod_index
