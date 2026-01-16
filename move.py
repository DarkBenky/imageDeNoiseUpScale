import os
import shutil

SRC = "/media/user/f7a503ec-b25c-41a5-9baa-d350714f613a/imageData"
DST = "/media/user/2TB/imageData"

os.makedirs(DST, exist_ok=True)

folders = sorted(
    f for f in os.listdir(SRC)
    if f.startswith("image_") and os.path.isdir(os.path.join(SRC, f))
)

i = 1
for folder in folders:
    new_name = f"image_{i:08d}"
    shutil.move(
        os.path.join(SRC, folder),
        os.path.join(DST, new_name)
    )
    i += 1
    if i % 100 == 0:
        print(f"Renamed {i} folders out of {len(folders)}")

print("Renaming completed.")
