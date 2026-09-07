"""
Shared file for data loading, IMU alignment, feature engineering and cross-validation

used by: data_analysis.ipynb, knn.py, svc.py, rforest.py (placeholder names)

Pipeline:
    raw imu.csv + annotation.csv
        -> load data
        -> align annotations
        -> window features
        -> get features and labels

"""



import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold



# Load Data
# ----------------

# Header row for raw imu_data.csv 
#column order in assignment description: 
# values are raw 16-bit signed integers representing:
#   accel: +/- 8g full-scale range
#   gyro: +/- 8000 dps full-scale range

IMU_COLUMNS = ["timestamp", "accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]



# load data and annotation files
def load_imu_data(path="a3_imu_data.csv"):
    return pd.read_csv(path, header=None, names=IMU_COLUMNS)


def load_annotations(path="a3_activity_annotations.csv"):
    df = pd.read_csv(path)
    df["start_time_sec"] = df["image"].str.extract(r"t=([\d.]+)").astype(float)
    return df[["start_time_sec", "label"]].sort_values("start_time_sec").reset_index(drop=True)




# unit conversion of 16-bit signed value to physical units
# put this here just in case - do we want values in the raw counts or g/dps?
ACCEL_SCALE_G = 8 / 32768
GYRO_SCALE_DPS = 8000 / 32768

def to_physical_units(imu_df):
    df = imu_df.copy()
    df[["accel_x", "accel_y", "accel_z"]] *= ACCEL_SCALE_G
    df[["gyro_x", "gyro_y", "gyro_z"]] *= GYRO_SCALE_DPS
    return df


