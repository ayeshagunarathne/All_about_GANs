#----------------------------------------------------------------------------------------------------------------------------
#TRAINING A SIMPLE GAN TO GENERATE HAND WRITEN DIGITS (0-9) - Trained using MNIST dataset
#----------------------------------------------------------------------------------------------------------------------------

#Importing the libraries that we need
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

torch.manual_seed(0)
def show_tensor_images(image_tensor, num_images=5, size=(1, 28, 28)):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in a uniform grid.
    '''
    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()


class Discriminator(nn.Module):
    def __init__(self, in_features=784, hidden_dim=128, out_features=1):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(in_features, hidden_dim * 4),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, out_features),

        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, noise, in_features=784, hidden_dim=128):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(noise, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 4, hidden_dim * 8),
            nn.BatchNorm1d(hidden_dim * 8),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 8, in_features),
            #MNIST dataset images are 28x28 gray scale images> so in features = 784
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.gen(x)


#Defining hyperparameters
device = "cuda" if torch.cuda.is_available() else "cpu"
criterion = nn.BCEWithLogitsLoss()
lr = 1e-5
z_dim = 64
display_step = 1000
batch_size = 128
num_epochs = 200

#intializing Generator and Discriminator
disc = Discriminator().to(device)
gen = Generator(z_dim).to(device)

transforms = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))
     ]
)
dataloader = DataLoader(
    MNIST(root="dataset/", download=True, transform=transforms),
    batch_size=batch_size,
    shuffle=True
)

disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)
gen_opt = torch.optim.Adam(disc.parameters(), lr=lr)
current_step = 0
mean_discriminator_loss = 0
mean_generator_loss = 0

for epoch in range(num_epochs):
    for real, _ in tqdm(dataloader):  #we dont use labels so we use _ here

        current_batch_size = len(real)

        real = real.view(current_batch_size, -1).to(device)

        #Train Discriminator
        Noise_vector = torch.randn(batch_size, z_dim).to(device)
        fake_images = gen(Noise_vector)

        fake_predictions = disc(fake_images.detach())
        fake_label = torch.zeros_like(fake_predictions)
        fake_loss = criterion(fake_predictions, fake_label)

        real_predictions = disc(real)
        real_labels = torch.ones_like(real_predictions)
        real_loss = criterion(real_predictions, real_labels)

        disc_loss = (fake_loss + real_loss) / 2
        disc.zero_grad()
        disc_loss.backward()
        disc_opt.step()

        #Training generator
        output_prediction = disc(fake_images)
        real_label_generator = torch.ones_like(output_prediction)
        gen_loss = criterion(output_prediction, real_label_generator)

        gen.zero_grad()
        gen_loss.backward()
        gen_opt.step()

        mean_discriminator_loss += disc_loss.item() / display_step
        mean_generator_loss += gen_loss.item() / display_step

        if current_step % display_step == 0 and current_step > 0:
            print(
                f"step{current_step}: Generator loss: {mean_generator_loss}, Discriminator loss : {mean_discriminator_loss}")
            fake_noise = torch.randn(current_batch_size, z_dim)
            fake = gen(fake_noise)
            show_tensor_images(fake)
            show_tensor_images(fake)
            show_tensor_images(real)
            mean_generator_loss = 0
            mean_discriminator_loss = 0
        current_step += 1
