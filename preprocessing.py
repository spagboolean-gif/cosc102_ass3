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




# Assign activity label to each IMU sample

def align_annotations_to_imu(imu_df, annotations_df):

    imu_df = imu_df.sort_values("timestamp").reset_index(drop=True)
    imu_df["timestamp"] = imu_df["timestamp"].astype("float64")
    imu_start = imu_df["timestamp"].min()

    annotations_df = annotations_df.copy()
    annotations_df["timestamp"] = (imu_start + annotations_df["start_time_sec"]).astype("float64")
    annotations_df = annotations_df.sort_values("timestamp").reset_index(drop=True)

    return pd.merge_asof(imu_df, annotations_df[["timestamp", "label"]], on="timestamp", direction="backward")



# Feature engineering
# -----------------------

ACCEL_COLS = ["accel_x", "accel_y", "accel_z"]
GYRO_COLS = ["gyro_x", "gyro_y", "gyro_z"]


def window_features(labelled_df, window_size_sec=1):

    df = labelled_df.copy()
    t0 = df["timestamp"].min()
    df["window_id"] = ((df["timestamp"] - t0) // window_size_sec).astype(int)

    rows = []
    for window_id, window_df in df.groupby("window_id"):
        features = {"window_id" : window_id}

        for col in ACCEL_COLS + GYRO_COLS:
            features[f"{col}_mean"] = window_df[col].mean()
            features[f"{col}_min"] = window_df[col].min()
            features[f"{col}_max"] = window_df[col].max()

    # Signal Magnitude Area
    features["accel_sma"] = window_df[ACCEL_COLS].abs.sum(axis=1).mean()
    features["gyro_sma"] = window_df[GYRO_COLS].abs().sum(axis=1).mean()

    # avg vector magnitude intensity
    features["avg_intensity"] = np.sqrt(window_df[ACCEL_COLS].pow(2).sum(axis=1)).mean()

    features["label"] = window_df["label"].mode().iloc[0]
    rows.append(features)

    return pd.DataFrame(rows)


# Everyone should call this first sso we all use the same aligned and labelled features
def get_features_and_labels(imu_path="a3_imu_data.csv", annotations_path="a3_activity_annotations.csv", window_size_sec=1):

    imu_df = load_imu_data(imu_path)
    annotations_df = load_annotations(annotations_path)
    labelled_df = align_annotations_to_imu(imu_df, annotations_df)
    feature_df = window_features(labelled_df, window_size_sec=window_size_sec)

    x = feature_df.drop(columns=["window_id", "label"])
    y = feature_df["label"]

    return x, y


def check_class_balance(y):
    return y.value_counts()



# Cross-validation
# -------------------


def get_cv_splitter(n_splits=5, random_state=42):

    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
