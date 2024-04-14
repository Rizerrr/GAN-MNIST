# код для продолжения обучения модели GAN
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm  # Модуль для создания полосы прогресса в циклах
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from PIL import Image


# Определение генератора
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# Определение дискриминатора
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


BATCH_SIZE = 8 # Количество элементов в батче
# Загрузка данных MNIST и инициализация DataLoader
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
dataset = MNIST(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Создание экземпляров генератора и дискриминатора
generator = Generator()
discriminator = Discriminator()

# Загрузка сохраненной модели и параметров
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator.to(device)
discriminator.to(device)

checkpoint = torch.load('path_to_save_model/name.pt')
generator.load_state_dict(checkpoint['generator_state_dict'])
discriminator.load_state_dict(checkpoint['discriminator_state_dict'])

# Загрузка оптимизаторов
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002)
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002)

optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])

# Продолжение обучения
criterion = nn.BCELoss()

num_epochs = 100 # Количество эпох

for epoch in range(num_epochs):
    for i, (real_images, _) in enumerate(tqdm(dataloader, desc=f'Training Epoch [{epoch+6}/{num_epochs}]', position=0, leave=True)):
        batch_size = real_images.size(0)
        real_images = real_images.view(batch_size, -1).to(device)

        # Update discriminator
        optimizer_D.zero_grad()
        z = torch.randn(batch_size, 100).to(device)
        fake_images = generator(z)
        fake_images = fake_images.to(device)

        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        real_outputs = discriminator(real_images)
        fake_outputs = discriminator(fake_images.detach())

        d_loss_real = criterion(real_outputs, real_labels)
        d_loss_fake = criterion(fake_outputs, fake_labels)

        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_D.step()

        # Update generator
        optimizer_G.zero_grad()
        fake_outputs = discriminator(fake_images)
        g_loss = criterion(fake_outputs, real_labels)
        g_loss.backward()
        optimizer_G.step()

    num_samples = 5
    generated_images = generator(torch.randn(num_samples, 100).to(device))
    generated_images = generated_images.view(num_samples, 1, 28, 28)
    generated_images = generated_images.cpu().detach()

    for j in range(num_samples):
        img = generated_images[j].squeeze().numpy()
        img = (img + 1) / 2  # Нормализация изображения к диапазону [0, 1]
        img = Image.fromarray((img * 255).astype('uint8'))
        img = img.resize((300, 300))  # Изменение размера до крупного
        img_path = f"path_to_save/{epoch+1}_sample{j + 1}.png"
        img.save(img_path, format="PNG")

    # Save the model at the end of each epoch
    torch.save({
        'epoch': epoch + 1,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict()
    }, f'path_to_save/gan_model_epoch{epoch+1}.pt')

