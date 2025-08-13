import os

root = "data"
splits = ["train", "val", "test"]
classes = ["NORMAL", "PNEUMONIA"]

for s in splits:
    print(f"\n[{s.upper()}]")
    for c in classes:
        path = os.path.join(root, s, c)
        if not os.path.isdir(path):
            print(f"⚠ Missing folder: {path}")
            continue
        count = sum(
            1 for f in os.listdir(path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        print(f"{c:10s} → {count} images")
