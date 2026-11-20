"""
dnbr.py - Lập bản đồ vùng cháy vụ cháy Hoàng Liên (19-22/2/2024) bằng dNBR
Dữ liệu: Sentinel-2 L2A, ảnh trước cháy 5/2/2024, ảnh sau cháy 6/3/2024
"""
import glob
import os
import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage
import matplotlib.pyplot as plt


PRE_DIR = "s2_0205"                 # thư mục giải nén ảnh 5/2
POST_DIR = "s2_0306"                # thư mục giải nén ảnh 6/3
FIRMS_CSV = "diem_chay_sach.csv"    # 58 điểm FIRMS đã lọc hôm trước
THRESHOLDS = [0.10, 0.20, 0.27]     # các ngưỡng dNBR đặt trước
BUFFER_M = 500                      # bán kính quanh điểm FIRMS (mét)
BANDS = ["B03", "B08", "B8A", "B12"]


def read_band(folder, band):
    """Tìm file có tên chứa `band` trong thư mục, mở ra, trả về bảng số."""
    paths = [p for p in glob.glob(os.path.join(folder, "**", "*.tif*"), recursive=True)
             if band in os.path.basename(p)]
    if len(paths) != 1:
        raise RuntimeError(f"Tìm thấy {len(paths)} file cho {band} trong {folder}: {paths}")
    with rasterio.open(paths[0]) as src:
        values = src.read(1).astype("float64")       # lớp 1: giá trị phản xạ
        if src.count >= 2:                           # lớp 2: dataMask (1 = có dữ liệu)
            has_data = src.read(2) > 0
        else:
            has_data = np.isfinite(values) & (values > 0)
        return values, has_data, src.transform, src.crs

pre, post = {}, {}          # pre["B8A"] là bảng B8A ngày 5/2, ...
transforms = []
valid = None                # bảng Đúng/Sai: điểm ảnh nào dùng được
for folder, store in [(PRE_DIR, pre), (POST_DIR, post)]:
    for b in BANDS:
        values, has_data, tf, crs = read_band(folder, b)
        store[b] = values
        transforms.append(tf)
        valid = has_data if valid is None else (valid & has_data)


shapes = {v.shape for v in list(pre.values()) + list(post.values())}
assert len(shapes) == 1, f"Các bảng khác kích thước: {shapes}"
assert all(t == transforms[0] for t in transforms), "Các file lệch tọa độ"
tf = transforms[0]
rows, cols = pre["B8A"].shape
print(f"Kích thước ảnh: {rows} hàng x {cols} cột")

def normalized_difference(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (a - b) / (a + b)
    out[~np.isfinite(out)] = np.nan     # chỗ chia cho 0 thì gán "không có giá trị"
    return out

#MẶT NƯỚC
ndwi_pre = normalized_difference(pre["B03"], pre["B08"])
ndwi_post = normalized_difference(post["B03"], post["B08"])
with np.errstate(invalid="ignore"):
    water = (ndwi_pre > 0) | (ndwi_post > 0)    # là nước ở MỘT TRONG HAI ngày
valid &= ~water
print(f"Số điểm ảnh nước bị che: {water.sum()}")

nbr_pre = normalized_difference(pre["B8A"], pre["B12"])
nbr_post = normalized_difference(post["B8A"], post["B12"])
dnbr = nbr_pre - nbr_post
dnbr[~valid] = np.nan
print(f"NBR trước cháy (trung vị): {np.nanmedian(nbr_pre[valid]):.3f}")
print(f"dNBR (trung vị toàn khung): {np.nanmedian(dnbr):.3f}")

# transform: tf.c = kinh độ mép trái, tf.f = vĩ độ mép trên,
#            tf.a = bề rộng 1 ô (độ), tf.e = chiều cao 1 ô (độ, âm vì đi từ trên xuống)
lon = tf.c + (np.arange(cols) + 0.5) * tf.a
lat = tf.f + (np.arange(rows) + 0.5) * tf.e
LON, LAT = np.meshgrid(lon, lat)                 # bảng kinh độ, vĩ độ của từng ô
m_per_deg_lat = 110_750
m_per_deg_lon = 111_320 * np.cos(np.radians(lat.mean()))
px_area_ha = (abs(tf.e) * m_per_deg_lat) * (tf.a * m_per_deg_lon) / 10_000
print(f"Diện tích một điểm ảnh: {px_area_ha:.4f} ha")



firms = pd.read_csv(FIRMS_CSV)
firms = firms[firms["sensor"] == "VIIRS"]
near_firms = np.zeros((rows, cols), bool)
for lat0, lon0 in zip(firms["latitude"], firms["longitude"]):
    dy = (LAT - lat0) * m_per_deg_lat
    dx = (LON - lon0) * m_per_deg_lon
    near_firms |= (dx**2 + dy**2) <= BUFFER_M**2
print(f"Số điểm VIIRS dùng để lọc: {len(firms)}")


eight_neighbors = np.ones((3, 3), bool)          # hai ô chạm nhau cả theo đường chéo là liền khối
results = []
for thr in THRESHOLDS:
    with np.errstate(invalid="ignore"):
        burned = dnbr >= thr                     # tất cả điểm ảnh vượt ngưỡng
    labels, n_patches = ndimage.label(burned, structure=eight_neighbors)   # đánh số từng mảng
    keep = np.unique(labels[burned & near_firms])                         # mảng nào chạm vùng FIRMS
    keep = keep[keep > 0]
    fire = np.isin(labels, keep)                 # chỉ giữ các mảng đó
    results.append({
        "nguong_dNBR": thr,
        "toan_khung_ha": round(burned.sum() * px_area_ha, 2),
        "vu_chay_ha": round(fire.sum() * px_area_ha, 2),
        "so_manh_giu_lai": len(keep),
        "so_manh_toan_khung": n_patches,
    })
    with rasterio.open(f"vung_chay_{thr:.2f}.tif", "w", driver="GTiff", height=rows, width=cols,
                       count=1, dtype="uint8", crs=crs, transform=tf) as dst:
        dst.write(fire.astype("uint8"), 1)

table = pd.DataFrame(results)
print("\nDIỆN TÍCH VÙNG CHÁY:")
print(table.to_string(index=False))
table.to_csv("dien_tich_vung_chay.csv", index=False)


with rasterio.open("dnbr.tif", "w", driver="GTiff", height=rows, width=cols, count=1,
                   dtype="float32", crs=crs, transform=tf, nodata=np.nan) as dst:
    dst.write(dnbr.astype("float32"), 1)

extent = [lon[0] - tf.a / 2, lon[-1] + tf.a / 2, lat[-1] + tf.e / 2, lat[0] - tf.e / 2]
plt.figure(figsize=(9, 8))
plt.imshow(dnbr, cmap="RdYlGn_r", vmin=-0.2, vmax=0.8, extent=extent)
plt.colorbar(label="dNBR")
plt.scatter(firms["longitude"], firms["latitude"], s=12, c="blue", label="Điểm VIIRS 19-22/2")
plt.legend()
plt.xlabel("Kinh độ")
plt.ylabel("Vĩ độ")
plt.title("dNBR vụ cháy Hoàng Liên (5/2/2024 - 6/3/2024)")
plt.savefig("dnbr.png", dpi=200, bbox_inches="tight")
print("\nĐã lưu: dnbr.tif, dnbr.png, vung_chay_*.tif, dien_tich_vung_chay.csv")