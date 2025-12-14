import pandas as pd
import numpy as np
import joblib
from scipy.interpolate import interp1d
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

# --- CONFIGURATION ---
TRAIN_FILE = "/app/output/train.csv"
MODEL_FILE = "/app/output/flag_classifier.joblib"
ENCODER_FILE = "/app/output/label_encoder.joblib"


def process_segment(group, n_steps=100):
    group = group.sort_values("_ts_ms")
    data = group[["open", "high", "low", "close"]].values

    x_old = np.linspace(0, 1, len(data))
    x_new = np.linspace(0, 1, n_steps)
    f = interp1d(x_old, data, axis=0, kind="linear")
    data_resampled = f(x_new)

    start_price = data_resampled[0, 0]
    if start_price == 0:
        start_price = 1e-9

    return ((data_resampled / start_price) - 1.0).flatten()


def prepare_dataset(df):
    X, y = [], []
    # Group by segment_id
    for _, group in df.groupby("segment_id"):
        if len(group) > 5:
            try:
                features = process_segment(group)
                X.append(features)
                y.append(group["label"].iloc[0])
            except Exception as e:
                print(f"Error processing segment: {e}")
    return np.array(X), np.array(y)


if __name__ == "__main__":
    print(f"Loading training data from {TRAIN_FILE}...")
    try:
        df_train = pd.read_csv(TRAIN_FILE)
    except FileNotFoundError:
        print("Error: train.csv not found")
        exit(1)

    print("Preprocessing training segments")
    X_train, y_train_raw = prepare_dataset(df_train)

    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)

    print(f"Training MLPClassifier on {len(X_train)} samples, with hidden layers 128, 64 and max_iter 5000")
    clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=5000, random_state=1)
    clf.fit(X_train, y_train)

    # Save models
    print("Saving model and encoder")
    joblib.dump(clf, MODEL_FILE)
    joblib.dump(le, ENCODER_FILE)
