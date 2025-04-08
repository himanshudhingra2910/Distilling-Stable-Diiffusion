import os
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

# 📂 Folders
teacher_dir = "data/teacher_generated"
student_dir = "data/generated_by_student/epoch_10"

# 🧪 Image indices to compare
num_images = 10
ssim_scores = []
image_ids = []

# 📸 Visual Grid Setup
fig, axs = plt.subplots(num_images, 2, figsize=(10, 4 * num_images))

for i in range(1, num_images + 1):
    teacher_img_path = os.path.join(teacher_dir, f"gen_0_{i}.png")
    student_img_path = os.path.join(student_dir, f"gen_{i}.png")

    if not os.path.exists(teacher_img_path) or not os.path.exists(student_img_path):
        print(f"Skipping: Missing {teacher_img_path} or {student_img_path}")
        continue

    # Read images
    teacher_img = cv2.imread(teacher_img_path)
    student_img = cv2.imread(student_img_path)

    if teacher_img is None or student_img is None:
        print(f"Skipping unreadable image pair {i}")
        continue

    # Convert to RGB and grayscale
    teacher_rgb = cv2.cvtColor(teacher_img, cv2.COLOR_BGR2RGB)
    student_rgb = cv2.cvtColor(student_img, cv2.COLOR_BGR2RGB)

    teacher_gray = cv2.cvtColor(teacher_rgb, cv2.COLOR_RGB2GRAY)
    student_gray = cv2.cvtColor(student_rgb, cv2.COLOR_RGB2GRAY)

    # Resize if needed
    if teacher_gray.shape != student_gray.shape:
        student_rgb = cv2.resize(student_rgb, (teacher_rgb.shape[1], teacher_rgb.shape[0]))
        student_gray = cv2.resize(student_gray, (teacher_gray.shape[1], teacher_gray.shape[0]))

    # Calculate SSIM
    score = ssim(teacher_gray, student_gray)
    ssim_scores.append(score)
    image_ids.append(f"Img {i}")

    # 🖼️ Add to subplot grid
    axs[i-1, 0].imshow(teacher_rgb)
    axs[i-1, 0].set_title(f"Teacher Image {i}")
    axs[i-1, 0].axis("off")

    axs[i-1, 1].imshow(student_rgb)
    axs[i-1, 1].set_title(f"Student Image {i}\nSSIM: {score:.3f}")
    axs[i-1, 1].axis("off")

plt.tight_layout()
plt.savefig("comparison_grid_with_ssim.png")
plt.show()

# 📊 Plot SSIM Bar Graph
plt.figure(figsize=(10, 4))
plt.bar(image_ids, ssim_scores, color='skyblue')
plt.ylim(0, 1)
plt.xlabel("Image Pair")
plt.ylabel("SSIM Score")
plt.title("SSIM: Teacher vs Student Generator")
plt.grid(True)
plt.tight_layout()
plt.savefig("ssim_score_barplot.png")
plt.show()

# 📈 Print average SSIM
avg_ssim = sum(ssim_scores) / len(ssim_scores)
print(f"\n✅ Average SSIM across {len(ssim_scores)} images: {avg_ssim:.3f}")
