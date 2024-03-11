import torch
from torch import nn
from torch import optim
import torchvision
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from models import Discriminator, Generator, initialize_weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 2e-4
BATCH_SIZE = 128
IMAGE_SIZE = 64
IMG_CHANNEL = 3
Z_DIM = 100
NUM_EPOCHS = 5
FEATURES_DISC = 64
FEATURES_GEN = 64

transforms = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(IMG_CHANNEL)],
            [0.5 for _ in range(IMG_CHANNEL)]
        ),
    ]
)

# dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms, download=True)
dataset = datasets.ImageFolder(root='celeb_dataset/', transform=transforms)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
gen = Generator(Z_DIM, IMG_CHANNEL, FEATURES_GEN).to(device)
disc = Discriminator(IMG_CHANNEL, FEATURES_DISC).to(device)
initialize_weights(gen)
initialize_weights(disc)

opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
criterion = nn.BCELoss()

fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)

# setting up the training
for epoch in range(NUM_EPOCHS):
    gen_loss, disc_loss = 0.0, 0.0
    for batch_idx, (real, _) in enumerate(loader):
        real = real.to(device)
        noise = torch.randn(BATCH_SIZE, Z_DIM, 1, 1).to(device)

        # train discriminator
        fake = gen(noise)
        disc_fake = disc(fake.detach()).view(-1)
        lossD_f = criterion(disc_fake, torch.zeros_like(disc_fake))
        disc_real = disc(real).view(-1)
        lossD_r = criterion(disc_real, torch.ones_like(disc_real))
        lossD = (lossD_f + lossD_r) / 2
        disc_loss += lossD
        disc.zero_grad()
        lossD.backward()
        opt_disc.step()

        # train generator
        disc_output = disc(fake).view(-1)
        lossG = criterion(disc_output, torch.zeros_like(disc_output))
        gen_loss += lossG
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()

    print(f"Epoch: {epoch}\nGenerator Training Loss: {gen_loss:.4f}, Discriminator Training loss: {disc_loss:.4f}")

torch.save(gen.state_dict(), "saved_models/generator.pth")
torch.save(disc.state_dict(), "saved_models/discriminator.pth")

