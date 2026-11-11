
import pandas as pd
from pathlib import Path


# 📁 ファイルの場所
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

INPUT_FILE = DATA_DIR / "temperature_data.csv"
OUTPUT_FILE = DATA_DIR / "processed_data.csv"


# 📊 必要な列
REQUIRED_COLUMNS = [
    "timestamp",
    "outside_temp",
    "inside_temp"
]


def load_data():
    """生データを読み込む"""

    print("🟡 CSVファイルを読み込み中...")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"入力ファイルが見つかりません: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print(f"🟡 読み込み件数: {len(df)}")

    # 🔍 必要な列が存在するか確認
    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"必要な列がありません: {missing_columns}"
        )

    return df


def clean_data(df):
    """データの形式を整え、欠損値を処理する"""

    print("🟡 データを整理中...")

    # 🕒 タイムスタンプを日時型に変換
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
        utc=True
    )

    # 🌡 温度を数値型に変換
    df["outside_temp"] = pd.to_numeric(
        df["outside_temp"],
        errors="coerce"
    )

    df["inside_temp"] = pd.to_numeric(
        df["inside_temp"],
        errors="coerce"
    )

    # 🔍 欠損値を含む行を削除
    before_count = len(df)

    df = df.dropna(
        subset=REQUIRED_COLUMNS
    ).copy()

    removed_count = before_count - len(df)

    print(f"🟡 欠損・変換不能データの削除: {removed_count}件")

    # 🕒 時刻順に並べ替え
    df = df.sort_values("timestamp").reset_index(drop=True)

    # 🔍 同じ時刻のデータがないか確認
    duplicate_count = df["timestamp"].duplicated().sum()

    if duplicate_count > 0:
        raise ValueError(
            f"同じタイムスタンプのデータが{duplicate_count}件あります"
        )

    print(f"🟢 整理後のデータ件数: {len(df)}")

    return df


def create_features(df):
    """機械学習に使用する特徴量を作成する"""

    print("🟡 特徴量を作成中...")

    # 🌡 外気温と槽内温度の差
    df["temp_diff"] = (
        df["outside_temp"] - df["inside_temp"]
    )

    # 🕒 前の測定時刻からの経過時間（秒）
    elapsed_seconds = (
        df["timestamp"].diff().dt.total_seconds()
    )

    # 🌡 外気温の変化速度（℃/秒）
    df["outside_rate"] = (
        df["outside_temp"].diff() / elapsed_seconds
    )

    # 🌡 槽内温度の変化速度（℃/秒）
    df["inside_rate"] = (
        df["inside_temp"].diff() / elapsed_seconds
    )

    # 🔍 変化速度を計算できない行を削除
    before_count = len(df)

    df = df.dropna(
        subset=["outside_rate", "inside_rate"]
    ).copy()

    removed_count = before_count - len(df)

    print(f"🟡 変化速度を計算できない行の削除: {removed_count}件")

    return df


def main():

    print("========== 前処理開始 ==========")

    # 📥 生データ読み込み
    df = load_data()

    # 🧹 データ整理
    df = clean_data(df)

    # ⚙️ 特徴量作成
    df = create_features(df)

    # 💾 CSV保存
    df.to_csv(OUTPUT_FILE, index=False)

    print("🟢 前処理完了！")
    print(f"🟢 保存先: {OUTPUT_FILE}")
    print(f"🟢 最終データ件数: {len(df)}")

    print("\n🟡 作成したデータの先頭5件")
    print(df.head())

    print("\n🟡 作成したデータの情報")
    print(df.info())


if __name__ == "__main__":
    main()