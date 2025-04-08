import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class GlassDefectDataset(Dataset):
    def __init__(self, image_folder, image_size=128):
        self.image_folder = image_folder
        self.image_files = [f for f in os.listdir(image_folder) if f.endswith(".png") or f.endswith(".jpg")]
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)  # Normalize to [-1, 1]
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_folder, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        return self.transform(image)
