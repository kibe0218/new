import requests
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

DATA_DIR.mkdir(exist_ok=True)

API_URL = "http://192.168.10.114:8080/api/v2"


LIMIT = 10
AFTER_ID = 9632

SENSOR_MAPPING = {
    "thermistor_outside": {
        "value": "outside_temp", 
        "id": "outside_id" 
    },
    "thermistor_inside": {
        "value": "inside_temp",
        "id": "inside_id"
    }
}


def fetch_temperature_data():
    print("🟡 API接続中...")
    try:
        response = requests.get(
            API_URL,
        params={
            "limit": LIMIT,
            "after_id": AFTER_ID
            },
            timeout=10 # 10秒以内に応答がなければタイムアウト
        )
        print("🟡 Request URL:", response.url)
        print("🟡 Status Code:", response.status_code)
        response.raise_for_status() # 成功したか確認
        data = response.json()
        print(f"🟢 {len(data)}件取得成功")
        return data
    
    except requests.exceptions.RequestException as e:
        print("🔴 API接続エラー:", e)
        return None


def convert_data(data):

    records = {}

    for item in data:
        data_id = item["id"]
        sensor_name = item["sensor"]
        if sensor_name not in SENSOR_MAPPING:
            continue
        timestamp = item["timestamp"]
        column = SENSOR_MAPPING[sensor_name]["value"]
        id_column = SENSOR_MAPPING[sensor_name]["id"]
        if timestamp not in records:
            records[timestamp] = {
                "timestamp": timestamp,
                "outside_temp": None,
                "inside_temp": None,
                "outside_id": None,
                "inside_id": None
            }
        records[timestamp][column] = item["value"]
        records[timestamp][id_column] = data_id


    df = pd.DataFrame(records.values()) #DataFrame型に変換
    if df.empty:
        print("🔴 有効なセンサーデータがありません")
        return df

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

def save_csv(df):
    filename = DATA_DIR / "temperature_data.csv"
    df.to_csv(filename, index=False)
    print("🟢 CSV保存完了:", filename)
    print(df.tail())

def main():
    data = fetch_temperature_data()
    if data is None:
        return

    df = convert_data(data)
    if df.empty:
        return

    save_csv(df)

if __name__ == "__main__":
    main()