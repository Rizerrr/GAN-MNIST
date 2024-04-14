import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm # Модуль для создания полосы прогресса в циклах
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


# Загрузка данных MNIST и инициализация DataLoader
# Среднее и стандартное отклонение на основе
#масштабированных данных набора MNIST (предварительно рассчитанные)

data_mean = 0.5
data_std = 0.5

# Преобразование входных изображений в тензоры и нормализация
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((data_mean,), (data_std,))
])

dataset = MNIST(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=4096, shuffle=True)

# Инициализация генератора, дискриминатора и оптимизаторов
device = torch.device('CUDA' if torch.cuda.is_available() else 'CPU')
generator = Generator().to(device)
discriminator = Discriminator().to(device)
criterion = nn.BCELoss()
lr = 0.0002
optimizer_G = optim.Adam(generator.parameters(), lr=lr)
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)
# Обучение модели GAN
print(torch.cuda.is_available())

num_epochs = 100 # Количество эпох
for epoch in range(num_epochs):
    for i, (real_images, _) in enumerate(tqdm(dataloader, desc=f'Training Epoch [{epoch+1}/{num_epochs}]', position=0, leave=True)):
        batch_size = real_images.size(0)
        real_images = real_images.view(batch_size, -1).to(device)

        # Обучение дискриминатора
        optimizer_D.zero_grad()
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # Дискриминатор обучается на реальных изображениях
        real_output = discriminator(real_images)
        d_loss_real = criterion(real_output, real_labels)

        # Генерация фейковых изображений и обучение дискриминатора на них
        z = torch.randn(batch_size, 100).to(device)  # Случайный шум
        fake_images = generator(z)
        fake_output = discriminator(fake_images.detach())  # Отключаем поток градиентов к генератору
        d_loss_fake = criterion(fake_output, fake_labels)

        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_D.step()

        # Обучение генератора
        optimizer_G.zero_grad()
        fake_output = discriminator(fake_images)
        g_loss = criterion(fake_output, real_labels)  # Генератор стремится обмануть дискриминатор
        g_loss.backward()
        optimizer_G.step()

    # Сохранение и удаление сгенерированных изображений
    num_samples = 5
    generated_images = generator(torch.randn(num_samples, 100).to(device))
    generated_images = generated_images.view(num_samples, 1, 28, 28)
    generated_images = generated_images.cpu().detach()

    # Сохранение сгенерированных изображений в файлы
    for j in range(num_samples):
        img = generated_images[j].squeeze().numpy()
        img = (img + 1) / 2  # Нормализация изображения к диапазону [0, 1]
        img = Image.fromarray((img * 255).astype('uint8'))
        img = img.resize((300, 300))  # Изменение размера до крупного
        img_path = f"exp/generated_images/generated_image_epoch{epoch + 1}_sample{j + 1}.png"
        img.save(img_path, format="PNG")

    # Сохранение состояния моделей
    torch.save({
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict()
    }, f'exp/gan_model_4096_epoch{epoch + 1}.pt')

