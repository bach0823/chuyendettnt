"""
Thử các định nghĩa mã ảnh gốc trên train, val, test.
File name format: YYYYMMDD_HHMMSS_X_Y.jpg hoặc tương tự.
"""
import os, glob, re
from collections import Counter

ROOT = r"d:\truong\SpecialSubjectTTNT\datasets\Crack500_ready"
if not os.path.exists(ROOT):
    ROOT = r"d:\truong\SpecialSubjectTTNT\SAGE_LITE\data\Crack500"

splits = ['train', 'val', 'test']

files_by_split = {}
for s in splits:
    d = os.path.join(ROOT, s, "images")
    if not os.path.exists(d):
        d = os.path.join(ROOT, s, "masks")
    flist = [os.path.basename(f) for f in glob.glob(os.path.join(d, "*.jpg")) + glob.glob(os.path.join(d, "*.png"))]
    flist = sorted(list(set(flist)))
    files_by_split[s] = flist
    print(f"[{s}] Total image/mask files: {len(flist)}")

# Xem 10 ví dụ tên file từ mỗi split
for s in splits:
    print(f"\nVí dụ 5 file {s}:")
    for f in files_by_split[s][:5]:
        print(f"  {f}")

def evaluate_definition(name, func):
    print(f"\n--- Định nghĩa: {name} ---")
    for s in splits:
        ids = [func(f) for f in files_by_split[s]]
        unique_ids = set(ids)
        print(f"  {s:5s}: {len(unique_ids):4d} unique IDs (tổng {len(ids)} crops, tb {len(ids)/len(unique_ids):.2f} crops/ID)")
        # In 3 ví dụ ID
        sample_ids = list(unique_ids)[:3]
        print(f"         Ví dụ ID: {sample_ids}")

# Định nghĩa 1: 2 phần đầu YYYYMMDD_HHMMSS (tách bằng _)
evaluate_definition("1. Tách bằng '_' lấy 2 phần đầu (date_time)", 
                    lambda f: "_".join(os.path.splitext(f)[0].split("_")[:2]))

# Định nghĩa 2: 3 phần đầu YYYYMMDD_HHMMSS_X (tách bằng _)
evaluate_definition("2. Tách bằng '_' lấy 3 phần đầu (date_time_X)", 
                    lambda f: "_".join(os.path.splitext(f)[0].split("_")[:3]))

# Định nghĩa 3: Chỉ phần date (YYYYMMDD)
evaluate_definition("3. Chỉ lấy phần ngày (YYYYMMDD)", 
                    lambda f: os.path.splitext(f)[0].split("_")[0])

# Định nghĩa 4: Regex tìm pattern YYYYMMDD_HHMMSS
def regex_datetime(f):
    m = re.match(r"(\d{8}_\d{6})", f)
    return m.group(1) if m else f

evaluate_definition("4. Regex YYYYMMDD_HHMMSS", regex_datetime)

# Định nghĩa 5: Tách bằng dấu '_' trừ phần crop cuối (nếu có 4 phần)
def drop_last_crop(f):
    parts = os.path.splitext(f)[0].split("_")
    if len(parts) >= 4:
        return "_".join(parts[:-2])  # date_time
    elif len(parts) == 3:
        return "_".join(parts[:-1])
    return parts[0]

evaluate_definition("5. Bỏ 2 tọa độ crop cuối (X, Y)", drop_last_crop)

# Kiểm tra phân bố số crop theo ảnh gốc của định nghĩa 1
print("\n=== Phân bố số crop mỗi ảnh gốc (Định nghĩa 1: YYYYMMDD_HHMMSS) ===")
for s in splits:
    c = Counter(["_".join(os.path.splitext(f)[0].split("_")[:2]) for f in files_by_split[s]])
    counts = list(c.values())
    print(f"{s:5s}: min={min(counts)}, max={max(counts)}, mean={sum(counts)/len(counts):.2f}, median={sorted(counts)[len(counts)//2]}")
    most_common = c.most_common(5)
    print(f"       Top 5 ảnh có nhiều crop nhất: {most_common}")
