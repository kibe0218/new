import requests
import pandas as pd
from pathlib import Path
import json

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
STATE_FILE = DATA_DIR / "last_id.json"

DATA_DIR.mkdir(exist_ok=True)

API_URL = "http://localhost:8080/api/v2"

LIMIT = 10000

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


def fetch_temperature_data(after_id):
    print("🟡 API接続中...")
    try:
        response = requests.get(
            API_URL,
            params={
                "limit": LIMIT,
                "after_id": after_id
            },
            timeout=10 # 10秒以内に応答がなければタイムアウト
        )
        print("🟡 Request URL:", response.url)
        print("🟡 Status Code:", response.status_code)
        response.raise_for_status() # 成功したか確認
        data = response.json()
        if data is None:
            print("🔴 データが取得できませんでした")
            data = []
        print(f"🟢 {len(data)}件取得成功")
        return data
    
    except requests.exceptions.RequestException as e:
        print("🔴 API接続エラー:", e)
        return None

def fetch_all_temperature_data(after_id):
    all_data = []
    while True:
        data = fetch_temperature_data(after_id)
        if data is None:
            return None
        if len(data) == 0:
            break
        all_data.extend(data)
        after_id = max(item["id"] for item in data)
        if len(data) < LIMIT:
            break
    return all_data

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
    csv_path = DATA_DIR / "temperature_data.csv"
    if not df.empty:
        df.to_csv(
            csv_path,
            mode="a",
            header=not csv_path.exists(),
            index=False
        )
        print("🟢 CSVへの保存が完了しました")
    else:
        print("🟡 保存するデータがありません")
    print(df.tail())

def load_last_id():
    if not STATE_FILE.exists():
        return 1000
    with open(STATE_FILE, "r") as f:
        state = json.load(f)
    return state["after_id"]

def save_last_id(last_id):
    with open(STATE_FILE, "w") as f:
        json.dump({"after_id": last_id}, f, indent=4)

def main():
    after_id = load_last_id()
    print(f"🟡 前回の取得位置: {after_id}")

    data = fetch_all_temperature_data(after_id)
    if data is None:
        print("🔴 データ取得に失敗しました")
        return
    elif len(data) == 0:
        print("🟡 新しいデーターはありません")
        return
    

    df = convert_data(data)
    if df.empty:
        return

    latest_id = max(item["id"] for item in data)

    save_csv(df)

    save_last_id(latest_id)

    print(f"🟡 取得位置を{latest_id}に更新しました")

if __name__ == "__main__":
    main()