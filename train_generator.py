import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from dataset_loader import GlassDefectDataset
from student_generator import StudentGenerator
import os

# Settings
EPOCHS = 50
BATCH_SIZE = 16
Z_DIM = 100
SAVE_DIR = "generated_by_student"
os.makedirs(SAVE_DIR, exist_ok=True)

# Device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Data
dataset = GlassDefectDataset("data/teacher_generated")
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model
model = StudentGenerator(z_dim=Z_DIM).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
criterion = torch.nn.L1Loss()

# Training Loop
for epoch in range(EPOCHS):
    for batch in dataloader:
        real_imgs = batch.to(device)
        noise = torch.randn(real_imgs.size(0), Z_DIM, 1, 1).to(device)

        fake_imgs = model(noise)

        loss = criterion(fake_imgs, real_imgs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {loss.item():.4f}")

    # Save some samples every 10 epochs
    if (epoch+1) % 10 == 0:
        save_path = os.path.join(SAVE_DIR, f"epoch_{epoch+1}")
        os.makedirs(save_path, exist_ok=True)

        save_image(fake_imgs, os.path.join(save_path, "gen_batch.png"), normalize=True)

