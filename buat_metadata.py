import csv
import os

root = "dataset"
output = "metadata.csv"

classes = {
    "fist": ("Fist", 1),
    "palm": ("Palm", 2),
    "victory": ("Victory", 3)
}

rows = []

for label, (gesture_name, mode) in classes.items():
    folder = os.path.join(root, label)

    if not os.path.isdir(folder):
        continue

    for filename in sorted(os.listdir(folder)):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            rows.append([
                filename,
                label,
                gesture_name,
                mode,
                os.path.join(folder, filename)
            ])

with open(output, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "label", "gesture", "mode", "path"])
    writer.writerows(rows)

print("Metadata selesai dibuat:", output)
print("Total gambar:", len(rows))
