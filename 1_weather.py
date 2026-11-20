"""Lấy thời tiết lịch sử theo giờ cho vụ cháy Hoàng Liên (19-22/2/2024) từ Open-Meteo."""
import requests
import pandas as pd

LAT, LON = 22.24626, 103.87037      # tâm 3 điểm VIIRS đầu tiên (vùng phát cháy)

params = {
    "latitude": LAT,
    "longitude": LON,
    "start_date": "2024-02-12",
    "end_date": "2024-02-24",
    "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,wind_direction_10m",
    "wind_speed_unit": "ms",          # m/s, đúng đơn vị mô hình dùng
    "timezone": "Asia/Bangkok",       # giờ Việt Nam
}
r = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params, timeout=30)
r.raise_for_status()                  # báo lỗi nếu máy chủ trả lỗi
data = r.json()

df = pd.DataFrame(data["hourly"]).rename(columns={
    "temperature_2m": "temp",
    "relative_humidity_2m": "humidity",
    "precipitation": "prcp",
    "wind_speed_10m": "wind_speed",
    "wind_direction_10m": "wind_deg",   # hướng gió THỔI TỪ, đúng quy ước của mô hình
})
df["time"] = pd.to_datetime(df["time"])
df.to_csv("research/thoi_tiet_hoang_lien.csv", index=False)

print(df[(df["time"] >= "2024-02-19 10:00") & (df["time"] <= "2024-02-20 12:00")].to_string(index=False))
print("Độ cao dùng cho dữ liệu thời tiết:", data.get("elevation"), "m")

df["ngay"] = df["time"].dt.date
tom_tat = df.groupby("ngay", as_index = False).agg(
    mua_mm=("prcp", "sum"),           # tổng lượng mưa trong ngày
    am_tb=("humidity", "mean"),       # độ ẩm trung bình
    am_min=("humidity", "min"),       # độ ẩm thấp nhất
    gio_max=("wind_speed", "max"),    # gió mạnh nhất
    nhiet_max=("temp", "max"),        # nhiệt độ cao nhất
).round(1)

tom_tat.to_csv("research/data_summary.csv", index=False)
print(tom_tat.to_string())

with open("research/grid_height.txt", "w", encoding="utf-8") as f:
    f.write(f"Nguon: Open-Meteo archive | toa do: {LAT}, {LON} | do cao o luoi: {data.get('elevation')} m\n")