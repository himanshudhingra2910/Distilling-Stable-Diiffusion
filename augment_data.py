import os
import cv2
import albumentations as A
from albumentations.augmentations.geometric.transforms import HorizontalFlip, VerticalFlip, Transpose
from albumentations.augmentations.transforms import (
    RandomRotate90, GaussNoise, RandomBrightnessContrast, MotionBlur, ShiftScaleRotate, Resize
)
from tqdm import tqdm

# Paths
INPUT_DIR = "data/teacher_generated"
OUTPUT_DIR = "data/augmented_generated"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define your augmentation pipeline
transform = A.Compose([
    RandomRotate90(p=0.5),
    HorizontalFlip(p=0.5),
    VerticalFlip(p=0.3),
    Transpose(p=0.3),
    GaussNoise(var_limit=(10.0, 50.0), p=0.4),
    RandomBrightnessContrast(p=0.5),
    MotionBlur(blur_limit=5, p=0.3),
    ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5),
    Resize(128, 128)
])

# Number of augmented versions per original image
AUG_PER_IMAGE = 25

# Process all images
for img_file in tqdm(os.listdir(INPUT_DIR)):
    img_path = os.path.join(INPUT_DIR, img_file)
    image = cv2.imread(img_path)

    if image is None:
        print(f"Skipping unreadable file: {img_path}")
        continue

    for i in range(AUG_PER_IMAGE):
        augmented = transform(image=image)["image"]
        output_path = os.path.join(OUTPUT_DIR, f"{img_file.split('.')[0]}_aug_{i}.png")
        cv2.imwrite(output_path, augmented)
