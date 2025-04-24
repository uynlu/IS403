import pandas as pd
from datetime import datetime
import json


def convert_str_to_datetime(dataframe: pd.DataFrame, format="%Y-%m-%d"):
    return pd.to_datetime(dataframe, format=format)


def convert_str_to_monthyear(dataframe: pd.DataFrame):
    return dataframe.dt.to_period("M")


def convert_quarter_to_datetime(quarter_str: str):
    quarter, year = quarter_str.split()
    year = int(year)

    if quarter == "Q1":
        return pd.Timestamp(year=year, month=1, day=1)
    elif quarter == "Q2":
        return pd.Timestamp(year=year, month=4, day=1)
    elif quarter == "Q3":
        return pd.Timestamp(year=year, month=7, day=1)
    elif quarter == "Q4":
        return pd.Timestamp(year=year, month=10, day=1)


def convert_datetime_to_quarteryear_str(date: datetime):
    quarter = (date.month - 1) // 3 + 1
    return f"Q{quarter} {date.year}"

    
def get_quarter_from_date(date: datetime):
    quarteryear_str = ""
    if date.month in [1, 2, 3]:
        quarteryear_str += "Q1"
    elif date.month in [4, 5, 6]:
        quarteryear_str += "Q2"
    elif date.month in [7, 8, 9]:
        quarteryear_str += "Q3"
    else:
        quarteryear_str += "Q4"

    quarteryear_str = quarteryear_str + " " + str(date.year)
    return quarteryear_str


def convert_to_float(dataframe):
    return dataframe.str.replace(",", "").astype(float)


def get_quarter(dataframe: pd.DataFrame):
    dataframe["Quarter"] = "Q" + dataframe["Kỳ"].astype(str) + " " + dataframe["Năm"].astype(str)
    return dataframe

def read_json_file(json_file_path: str):
    with open(json_file_path, "r") as file:
        loaded_file = json.load(file)
    return loaded_file
