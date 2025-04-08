import os

img_dir = "glass_yolo_dataset/images/val"
label_dir = "glass_yolo_dataset/labels/val"
os.makedirs(label_dir, exist_ok=True)

for fname in os.listdir(img_dir):
    if fname.endswith(".png") or fname.endswith(".jpg"):
        name = os.path.splitext(fname)[0]
        with open(os.path.join(label_dir, f"{name}.txt"), "w") as f:
            f.write("0 0.5 0.5 0.4 0.4\n")  # dummy centered bbox
