"""
2_luoi.py - Đưa vùng cháy thật từ bản đồ dNBR (ô 10 m) về lưới mô hình 540 x 540 (ô ~30 m).
Một ô mô hình tính là "cháy" nếu ít nhất 50% diện tích ô 10 m bên trong nó cháy.
Soạn với hỗ trợ của Claude (Anthropic).
"""
import importlib
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
import matplotlib.pyplot as plt

mo = importlib.import_module("1_MODULE")     # chỉ lấy kích thước lưới, không chạy mô hình

# ---------- THIẾT LẬP ----------
LAT_C, LON_C = 22.240, 103.875               # tâm ROI (giống mô hình)
DNBR_TIF = "research/dnbr.tif"                        # bản đồ dNBR (ngoài đa giác là NaN)
VUNG_CHAY_TIF = "vung_chay_{}.tif"           # vùng cháy theo từng ngưỡng, do dnbr.py tạo
NGUONG = ["0.10", "0.20", "0.27"]
OUT_NPZ = "research/vung_chay_luoi.npz"
OUT_PNG = "research/vung_chay_luoi.png"

# ---------- 1. Mô tả lưới mô hình: 4 mép khung và kích thước ô ----------
south, north = LAT_C - mo.LAT_SPAN_DEG / 2, LAT_C + mo.LAT_SPAN_DEG / 2
west, east = LON_C - mo.LON_SPAN_DEG / 2, LON_C + mo.LON_SPAN_DEG / 2
luoi_tf = from_bounds(west, south, east, north, mo.GRID_SHAPE[1], mo.GRID_SHAPE[0])
dy_m, dx_m, o_ha = mo.compute_projected_pixel_scales(LAT_C)
print(f"Một ô mô hình: {dy_m:.1f} m x {dx_m:.1f} m = {o_ha:.4f} ha")


# ---------- 2. Hàm chuyển một bản đồ 10 m sang lưới mô hình ----------
def ve_luoi(path, lay_vung_co_du_lieu=False):
    """Trả về bảng 540 x 540: mỗi ô là TỶ LỆ (0 đến 1) diện tích ô 10 m bên trong có giá trị Đúng.
    - lay_vung_co_du_lieu=False: Đúng = ô 10 m bị cháy (file vung_chay_*.tif)
    - lay_vung_co_du_lieu=True:  Đúng = ô 10 m có dữ liệu, tức nằm trong đa giác (file dnbr.tif)"""
    with rasterio.open(path) as src:
        a = src.read(1)
        dung = np.isfinite(a) if lay_vung_co_du_lieu else (a > 0)
        ty_le = np.zeros(mo.GRID_SHAPE, np.float32)
        reproject(
            source=dung.astype(np.float32), destination=ty_le,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=luoi_tf, dst_crs="EPSG:4326",
            resampling=Resampling.average,       # lấy trung bình = tỷ lệ phần cháy
        )
    return ty_le


# ---------- 3. Vùng có dữ liệu quan sát (trong đa giác) ----------
ket_qua = {"footprint": ve_luoi(DNBR_TIF, lay_vung_co_du_lieu=True) >= 0.5}
print(f"Số ô có dữ liệu quan sát: {int(ket_qua['footprint'].sum())}")

# ---------- 4. Vùng cháy thật theo từng ngưỡng ----------
print("\nNgưỡng | số ô cháy | diện tích (ô >= 50%) | diện tích theo tỷ lệ")
for ng in NGUONG:
    ty_le = ve_luoi(VUNG_CHAY_TIF.format(ng))
    chay = ty_le >= 0.5                      # quy tắc 50%
    ket_qua[f"obs_{ng.replace('.', '')}"] = chay
    print(f"{ng}   | {int(chay.sum()):6d}    | {chay.sum() * o_ha:7.1f} ha        | {ty_le.sum() * o_ha:7.1f} ha")

np.savez_compressed(OUT_NPZ, **ket_qua)
print(f"\nĐã lưu {OUT_NPZ} (các bảng: {', '.join(ket_qua)})")

# ---------- 5. Hình kiểm tra: phóng vào quanh vùng cháy ----------
obs = ket_qua["obs_020"]
r, c = np.nonzero(obs)
r1, r2 = max(r.min() - 20, 0), min(r.max() + 20, mo.GRID_SHAPE[0])
c1, c2 = max(c.min() - 20, 0), min(c.max() + 20, mo.GRID_SHAPE[1])
plt.figure(figsize=(8, 7))
plt.imshow(ket_qua["footprint"][r1:r2, c1:c2], cmap="Greys", alpha=0.25)   # xám: trong đa giác
plt.imshow(np.ma.masked_where(~obs[r1:r2, c1:c2], obs[r1:r2, c1:c2]), cmap="autumn")  # đỏ: cháy
plt.title(f"Vùng cháy thật trên lưới mô hình (ngưỡng 0,20) - hàng {r1}-{r2}, cột {c1}-{c2}")
plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
print(f"Đã lưu {OUT_PNG}")