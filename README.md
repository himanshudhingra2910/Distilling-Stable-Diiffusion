#  Distilling Stable Diffusion using Generative AI (Low-Resource Proof of Concept)

This project presents a lightweight, low-resource pipeline for generating synthetic cracked-glass images using Generative AI and training an object detection model (YOLOv5) entirely on this synthetic data — all done **on a CPU-only setup** (Mac M2, no GPU required).

Despite low image fidelity from the distilled student generator, the results show that structural relevance is sufficient to train a working fault detector, demonstrating that **realism isn't always required for functional training**.

---

## Key Highlights

-  **Stable Diffusion** used to generate high-quality cracked-glass images (teacher model)
-  **Distilled Student Generator** trained with L1 loss to mimic teacher images on low compute
-  **SSIM Comparison** shows structural similarity despite poor visual quality
-  **YOLOv5 trained** on student-generated images with dummy labels for defect detection
-  **Mac M2 CPU-only** implementation: no GPU used

---

## Tech Stack

- Python 3.9
- PyTorch
- Hugging Face `diffusers` (Stable Diffusion)
- OpenCV, Matplotlib
- YOLOv5 (Ultralytics)
- Mac M2, CPU

---


---

## 📈 Results

| Metric                  | Value      |
|-------------------------|------------|
| SSIM (avg Student vs Teacher) | ~0.108 |
| YOLOv5 Precision        | 1.00       |
| YOLOv5 Recall (peak)    | ~0.53      |
| YOLOv5 mAP@0.5          | ~0.32      |
| Training Hardware       | Mac M2 (CPU-only) |

---

## 📊 Visuals

### Student vs Teacher Image Comparison with SSIM  
![SSIM Comparison Grid](./ssim_analysis/comparison_grid_with_ssim.png)

### YOLOv5 Training Results  
![YOLOv5 Results](./yolov5_training/results.png)

### Validation Predictions  
![Predicted Boxes](./yolov5_training/val_batch0_pred.jpg)

---

## 📄 Research Paper

Read the full research write-up here:  
📎 `https://drive.google.com/file/d/1xwHcW-6fOl1_gdFRJ-175DyfG-Amwl_8/view?usp=sharing`

---

## 🚀 How to Run

### 1. Train the Student Generator
```bash
python train_generator.py
