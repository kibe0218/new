import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


# 📁 ファイルの場所
BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "data" / "processed_data.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_FILE = MODEL_DIR / "ocsvm_model.joblib"
SCALER_FILE = MODEL_DIR / "scaler.joblib"


# 📊 学習に使用する特徴量
FEATURE_COLUMNS = [
    "outside_temp",
    "inside_temp",
    "temp_diff",
    "outside_rate",
    "inside_rate"
]


def load_data():
    """加工済みデータを読み込む"""

    print("🟡 学習データを読み込み中...")

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"データファイルがありません: {DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)

    print(f"🟢 読み込み件数: {len(df)}")

    # 🔍 必要な特徴量が存在するか確認
    missing_columns = [
        col for col in FEATURE_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"必要な列がありません: {missing_columns}"
        )

    X = df[FEATURE_COLUMNS].apply(
        pd.to_numeric,
        errors="coerce"
    )

    # 🔍 欠損値・無限値を確認
    if not np.isfinite(X.to_numpy()).all():
        raise ValueError(
            "特徴量に欠損値または無限値があります"
        )

    return X


def train_model(X):
    """ScalerとOne-Class SVMを学習する"""

    print("🟡 データを標準化中...")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("🟢 標準化完了")

    # 🤖 One-Class SVM
    print("🟡 One-Class SVMを学習中...")

    model = OneClassSVM(
        kernel="rbf",
        gamma="scale",
        nu=0.05
    )

    model.fit(X_scaled)

    print("🟢 学習完了")

    return model, scaler


def save_model(model, scaler):
    """学習済みモデルを保存する"""

    MODEL_DIR.mkdir(exist_ok=True)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)

    print("🟢 モデル保存完了")
    print(f"モデル: {MODEL_FILE}")
    print(f"Scaler: {SCALER_FILE}")


def main():

    print("========== 学習開始 ==========")

    # 📥 データ読み込み
    X = load_data()

    # 🤖 学習
    model, scaler = train_model(X)

    # 🔍 学習データに対する判定結果
    predictions = model.predict(
        scaler.transform(X)
    )

    normal_count = np.sum(predictions == 1)
    anomaly_count = np.sum(predictions == -1)

    print("\n========== 学習結果 ==========")
    print(f"学習データ件数: {len(X)}")
    print(f"正常判定: {normal_count}")
    print(f"異常判定: {anomaly_count}")

    # 💾 保存
    save_model(model, scaler)

    print("\n🟢 すべての処理が完了しました")


if __name__ == "__main__":
    main()
