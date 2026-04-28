import os
import re
import numpy as np
import SimpleITK as sitk
from PIL import Image
from collections import defaultdict

# ======================
# 路径设置
# ======================
pred_dir = r"/home/wusi/deeplabv3-plus-pytorch/output"
ref_root = r"/home/wusi/SAMdata/20250711_GTVp/datanii/test_nii"
save_dir = r"/home/wusi/deeplabv3-plus-pytorch/nii_output"
os.makedirs(save_dir, exist_ok=True)

# 匹配文件名：p_xxx_sliceN.png
pattern = re.compile(r"^(p_\d+)_slice(\d+)\.(jpg|png|jpeg|tif)$", re.IGNORECASE)

# 收集每个患者的预测切片
patient_slices = defaultdict(list)
for f in os.listdir(pred_dir):
    m = pattern.match(f)
    if m:
        pid = m.group(1)
        sid = int(m.group(2))
        patient_slices[pid].append((sid, f))

print(f"共检测到 {len(patient_slices)} 个患者预测结果。")

# ======================
# 主循环：逐患者生成 nii.gz
# ======================
for pid, slice_files in patient_slices.items():
    ref_path = os.path.join(ref_root, pid, "image.nii.gz")
    if not os.path.exists(ref_path):
        print(f"⚠️ 未找到CT: {ref_path}")
        continue

    ref_img = sitk.ReadImage(ref_path)
    ref_arr = sitk.GetArrayFromImage(ref_img)
    depth, h, w = ref_arr.shape

    volume = np.zeros((depth, h, w), dtype=np.uint8)

    # 按切片号填充，不按顺序排序
    for sid, fname in slice_files:
        if sid >= depth:
            continue
        img = np.array(Image.open(os.path.join(pred_dir, fname)).convert("L"))
        volume[sid] = (img > 127).astype(np.uint8)

    sitk_img = sitk.GetImageFromArray(volume)
    sitk_img.CopyInformation(ref_img)

    # 直接取患者编号
    pid_num = int(re.search(r'p_(\d+)', pid).group(1))
    save_name = f"GTVp_{pid_num:03d}.nii.gz"
    save_path = os.path.join(save_dir, save_name)
    sitk.WriteImage(sitk_img, save_path)

    print(f"✅ 已保存: {save_name} （来自 {pid}，层数={volume.shape[0]}）")

print("\n全部患者处理完成。")
