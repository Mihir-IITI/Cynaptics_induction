import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

from PIL import Image
import os

#hyperparameters
batch_size = 32
learning_rate = 0.0002
epochs = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
alpha = 0.2

#from GAN model
class ImageDiscriminator(nn.Module):
    def __init__(self):
        super(ImageDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(alpha, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(alpha, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(alpha, inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(alpha, inplace=True),

            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 1024),
            nn.LeakyReLU(alpha, inplace=True),
            nn.Linear(1024, 1),
        )

    def forward(self, x):
        return self.model(x)



#determines how images transform
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


dataset = datasets.ImageFolder(
    root="/kaggle/input/set-of-images/New_Data/New_Data", 
    transform=transform,
)

#also assigns labels, AI=0 Real = 1 (alphabetically)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#creating model object, loss function type, optimizer
model = ImageDiscriminator().to(device)
criterion = nn.BCEWithLogitsLoss() #Sigmoid on BCE loss
optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.999)) #adam


#training
for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    for images, labels in data_loader:

        #loading data
        images, labels = images.to(device), labels.float().unsqueeze(1).to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(data_loader):.4f}")

print("Training complete!")

#saving
torch.save(model.state_dict(), "image_discriminator.pth")
