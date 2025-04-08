import cv2
import os

# Path to your student grid image
grid_path = "data/generated_by_student/sample_epoch_10.png"
output_dir = "data/generated_by_student/epoch_10"
os.makedirs(output_dir, exist_ok=True)

# Load image
image = cv2.imread(grid_path)

# Number of rows and columns in grid
rows, cols = 4, 4  # change if different
h, w, _ = image.shape
crop_h, crop_w = h // rows, w // cols

# Split and save each sub-image
count = 1
for i in range(rows):
    for j in range(cols):
        crop = image[i * crop_h:(i + 1) * crop_h, j * crop_w:(j + 1) * crop_w]
        out_path = os.path.join(output_dir, f"gen_{count}.png")
        cv2.imwrite(out_path, crop)
        print(f"Saved: {out_path}")
        count += 1
