
import pandas as pd
import numpy as np
import joblib

from pathlib import Path


# 📁 ファイルの場所
BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "data" / "processed_data.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_FILE = MODEL_DIR / "ocsvm_model.joblib"
SCALER_FILE = MODEL_DIR / "scaler.joblib"

OUTPUT_FILE = BASE_DIR / "data" / "prediction_results.csv"


# 🕒 判定対象期間（日本時間）
# Pゲイン0.5に変更した後の15分間
START_TIME = pd.Timestamp(
    "2026-11-12 16:45:00",
    tz="Asia/Tokyo"
)

END_TIME = pd.Timestamp(
    "2026-11-12 17:00:00",
    tz="Asia/Tokyo"
)


# 📊 学習時と同じ特徴量
FEATURE_COLUMNS = [
    "outside_temp",
    "inside_temp",
    "temp_diff",
    "outside_rate",
    "inside_rate"
]


def load_model():
    """保存済みモデルとScalerを読み込む"""

    print("🟡 学習済みモデルを読み込み中...")

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"モデルがありません: {MODEL_FILE}"
        )

    if not SCALER_FILE.exists():
        raise FileNotFoundError(
            f"Scalerがありません: {SCALER_FILE}"
        )

    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)

    print("🟢 モデル読み込み完了")

    return model, scaler


def load_data():
    """判定対象期間のデータを読み込む"""

    print("\n🟡 判定データを読み込み中...")

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"データがありません: {DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)

    print(f"🟢 全データ件数: {len(df)}")

    # 🔍 必要な列を確認
    required_columns = [
        "timestamp"
    ] + FEATURE_COLUMNS

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"必要な列がありません: {missing_columns}"
        )

    # 🕒 timestampをUTCに変換
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce"
    )

    if df["timestamp"].isna().any():
        raise ValueError(
            "timestampに不正な値があります"
        )

    # 🌡 特徴量を数値型に変換
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].apply(
        pd.to_numeric,
        errors="coerce"
    )

    # 🔍 欠損値を確認
    before_count = len(df)

    df = df.dropna(
        subset=FEATURE_COLUMNS
    ).copy()

    removed_count = before_count - len(df)

    print(
        f"🟡 欠損データの削除: {removed_count}件"
    )

    # 🕒 日本時間をUTCに変換
    start_utc = START_TIME.tz_convert("UTC")
    end_utc = END_TIME.tz_convert("UTC")

    # 🕒 指定した15分間だけ抽出
    df = df[
        (df["timestamp"] >= start_utc) &
        (df["timestamp"] < end_utc)
    ].copy()

    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    print("\n========== 判定対象期間 ==========")
    print(f"開始: {START_TIME}")
    print(f"終了: {END_TIME}")
    print(f"UTC開始: {start_utc}")
    print(f"UTC終了: {end_utc}")
    print(f"判定対象件数: {len(df)}")

    if len(df) == 0:
        raise ValueError(
            "指定期間内に判定対象データがありません"
        )

    # 🔍 欠損値・無限値を確認
    X = df[FEATURE_COLUMNS]

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError(
            "特徴量に欠損値または無限値があります"
        )

    print(
        f"最初のデータ: {df['timestamp'].iloc[0]}"
    )

    print(
        f"最後のデータ: {df['timestamp'].iloc[-1]}"
    )

    return df


def predict_data(df, model, scaler):
    """保存済みモデルで異常検知する"""

    print("\n🟡 異常検知を実行中...")

    X = df[FEATURE_COLUMNS]

    # 📊 学習時のScalerで標準化
    X_scaled = scaler.transform(X)

    # 🤖 One-Class SVMによる判定
    predictions = model.predict(X_scaled)

    # 📈 判定スコア
    scores = model.decision_function(X_scaled)

    # 📊 判定結果を追加
    result = df.copy()

    result["prediction"] = predictions

    result["判定"] = np.where(
        predictions == 1,
        "正常",
        "異常"
    )

    result["decision_score"] = scores

    print("🟢 異常検知完了")

    return result


def show_results(result):
    """判定結果を集計・表示する"""

    normal_count = int(
        (result["prediction"] == 1).sum()
    )

    anomaly_count = int(
        (result["prediction"] == -1).sum()
    )

    total_count = len(result)

    normal_rate = (
        normal_count / total_count * 100
    )

    anomaly_rate = (
        anomaly_count / total_count * 100
    )

    print("\n========== 異常検知結果 ==========")

    print(f"判定データ件数: {total_count}")

    print(
        f"正常判定: {normal_count}件 "
        f"({normal_rate:.2f}%)"
    )

    print(
        f"異常判定: {anomaly_count}件 "
        f"({anomaly_rate:.2f}%)"
    )

    print("\n========== 判定結果の先頭10件 ==========")

    print(
        result[
            [
                "timestamp",
                "outside_temp",
                "inside_temp",
                "判定",
                "decision_score"
            ]
        ].head(10).to_string(index=False)
    )


def save_results(result):
    """判定結果をCSVに保存する"""

    result.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n🟢 判定結果を保存しました")
    print(f"保存先: {OUTPUT_FILE}")


def main():

    print("========== 異常検知開始 ==========")

    # 📥 学習済みモデル読み込み
    model, scaler = load_model()

    # 📥 判定データ読み込み
    df = load_data()

    # 🤖 異常検知
    result = predict_data(
        df,
        model,
        scaler
    )

    # 📊 結果表示
    show_results(result)

    # 💾 結果保存
    save_results(result)

    print("\n🟢 すべての処理が完了しました")


if __name__ == "__main__":
    main()
