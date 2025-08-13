import kagglehub
import shutil
import os

print("📥 Downloading dataset from Kaggle...")
path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
print(f"✅ Downloaded to: {path}")

# The dataset folder contains: chest_xray/train, chest_xray/test, chest_xray/val
dest = "data"
os.makedirs(dest, exist_ok=True)

src_folder = os.path.join(path, "chest_xray")
for folder in ["train", "val", "test"]:
    src = os.path.join(src_folder, folder)
    dst = os.path.join(dest, folder)
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    print(f"📂 Copied {folder} → {dst}")

print("🎯 Dataset ready in 'data/' folder")
