import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# --- CONFIGURATION ---
TEST_DATA_FILE = "/app/output/test.csv"
MODEL_FILE = "/app/output/flag_classifier.joblib"
ENCODER_FILE = "/app/output/label_encoder.joblib"
FIGURE_OUTPUT = "/app/output/binary_confusion_matrix.png"


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
    X, y, ids = [], [], []
    for seg_id, group in df.groupby("segment_id"):
        if len(group) > 5:
            try:
                features = process_segment(group)
                X.append(features)
                y.append(group["label"].iloc[0])
                ids.append(seg_id)
            except Exception:
                pass
    return np.array(X), np.array(y), ids


if __name__ == "__main__":

    clf = joblib.load(MODEL_FILE)
    le = joblib.load(ENCODER_FILE)
    df_test = pd.read_csv(TEST_DATA_FILE)

    print(f"Preprocessing {df_test['segment_id'].nunique()} test segments...")
    X_test, y_test_raw, seg_ids = prepare_dataset(df_test)

    if len(X_test) == 0:
        print("No valid segments found in test file.")
        exit()

    # Handle potentially unseen labels during transformation
    valid_mask = np.isin(y_test_raw, le.classes_)
    if not np.all(valid_mask):
        print(f"Dropping {np.sum(~valid_mask)} segments with unknown labels.")
        X_test = X_test[valid_mask]
        y_test_raw = y_test_raw[valid_mask]

    y_test_encoded = le.transform(y_test_raw)

    print("\nRunning Predictions...")
    y_pred_encoded = clf.predict(X_test)
    y_pred_labels = le.inverse_transform(y_pred_encoded)

    print("=" * 40)
    print("CLASSIFICATION REPORT")
    print("=" * 40)
    print(classification_report(y_test_raw, y_pred_labels))

    # Bullish vs Bearish
    print("=" * 40)
    print("BINARY (DIRECTIONAL) REPORT")
    print("=" * 40)

    y_test_bin = np.array(list(map(lambda x: x[:7], y_test_raw)))
    y_pred_bin = np.array(list(map(lambda x: x[:7], y_pred_labels)))

    print(classification_report(y_test_bin, y_pred_bin))

    print("Binary Confusion Matrix:")
    cm = confusion_matrix(y_test_bin, y_pred_bin, labels=["Bullish", "Bearish"])

    # text output
    print(f"{'':<10} | {'Pred Bull':<10} | {'Pred Bear':<10}")
    print("-" * 36)
    print(f"{'True Bull':<10} | {cm[0][0]:<10} | {cm[0][1]:<10}")
    print(f"{'True Bear':<10} | {cm[1][0]:<10} | {cm[1][1]:<10}")

    # figures
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Bullish", "Bearish"])
    disp.plot(cmap="Blues")
    plt.title("Directional Confusion Matrix")
    plt.savefig(FIGURE_OUTPUT)
