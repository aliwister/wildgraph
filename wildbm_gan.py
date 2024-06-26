import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import pandas as pd
import pdb


class CustomDataset(Dataset):
    def __init__(self, df, tokenizer=None):
        self.data = df
        self.mean = np.mean(df)
        self.std = np.std(df)
        print(self.mean, self.std)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        #print(self.data[idx])
        sample = torch.tensor(self.data[idx]).float()
        normalized_sample = (sample - self.mean) / self.std
        return normalized_sample
"""
FILE ='./geese.csv'
columns = ['idx', 'label', 'location.lat', 'location.long']
df = pd.read_csv(FILE, usecols=columns)
real = df[['label', 'location.lat', 'location.long']].groupby('label')
group_arrays = [group[['location.lat', 'location.long']].to_numpy() for _, group in real]
train_array = np.array(group_arrays)
dataset = CustomDataset(train_array)
X_train = dataset

train_loader = DataLoader(X_train, batch_size=8, num_workers=8)
"""


PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
BATCH_SIZE = 256 if torch.cuda.is_available() else 64
NUM_WORKERS = int(os.cpu_count() / 2)

class GAN(L.LightningModule):
    def __init__(
        self,
        channels,
        width,
        latent_dim: int = 100,
        lr: float = 0.0002,
        b1: float = 0.5,
        b2: float = 0.999,
        batch_size: int = BATCH_SIZE,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        # networks
        data_shape = (channels, width)
        self.generator = Generator(latent_dim=self.hparams.latent_dim, img_shape=data_shape)
        self.discriminator = Discriminator(img_shape=data_shape)

        self.validation_z = torch.randn(8, self.hparams.latent_dim)

        self.example_input_array = torch.zeros(2, self.hparams.latent_dim)

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch):
        imgs  = batch

        optimizer_g, optimizer_d = self.optimizers()

        # sample noise
        z = torch.randn(imgs.shape[0], self.hparams.latent_dim)
        z = z.type_as(imgs)

        # train generator
        # generate images
        self.toggle_optimizer(optimizer_g)
        self.generated_imgs = self(z)

        # log sampled images
        sample_imgs = self.generated_imgs[:6]
        grid = torchvision.utils.make_grid(sample_imgs)
        #self.logger.experiment.add_image("generated_images", grid, 0)

        # ground truth result (ie: all fake)
        # put on GPU because we created this tensor inside training_loop
        valid = torch.ones(imgs.size(0), 1)
        valid = valid.type_as(imgs)

        # adversarial loss is binary cross-entropy
        g_loss = self.adversarial_loss(self.discriminator(self(z)), valid)
        self.log("g_loss", g_loss, prog_bar=True)
        self.manual_backward(g_loss)
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)

        # train discriminator
        # Measure discriminator's ability to classify real from generated samples
        self.toggle_optimizer(optimizer_d)

        # how well can it label as real?
        valid = torch.ones(imgs.size(0), 1)
        valid = valid.type_as(imgs)

        real_loss = self.adversarial_loss(self.discriminator(imgs), valid)

        # how well can it label as fake?
        fake = torch.zeros(imgs.size(0), 1)
        fake = fake.type_as(imgs)

        fake_loss = self.adversarial_loss(self.discriminator(self(z).detach()), fake)

        # discriminator loss is the average of these
        d_loss = (real_loss + fake_loss) / 2
        self.log("d_loss", d_loss, prog_bar=True)
        self.manual_backward(d_loss)
        optimizer_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []

    def on_training_epoch_end(self):
        z = self.validation_z.type_as(self.generator.model[0].weight)

        # log sampled images
        sample_imgs = self(z)
        grid = torchvision.utils.make_grid(sample_imgs)
        digit = sample_imgs[:1].detach().cpu().reshape(28, 28) # reshape vector to 2d array
        plt.imshow(digit, cmap='gray')
        plt.axis('off')
        plt.show()
        #self.logger.experiment.add_image("generated_images", grid, self.current_epoch)

class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super().__init__()
        self.img_shape = img_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh(),
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img

class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity

def run_gan(df, epochs, seq_length):
    real = df[['label', 'location.lat', 'location.long']].groupby('label')
    group_arrays = [group[['location.lat', 'location.long']].to_numpy() for _, group in real]
    train_array = np.array(group_arrays)
    dataset = CustomDataset(train_array)
    X_train = dataset

    train_loader = DataLoader(X_train, batch_size=8, num_workers=8)

    model = GAN(*(seq_length, 2))
    trainer = L.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=80,
    )
    trainer.fit(model, train_loader)

    SAMPLES = 60
    z = torch.randn(SAMPLES, 100)
    generated_imgs = model.generator(z)
    reverse_std = generated_imgs.reshape(SAMPLES,-1)*dataset.std + dataset.mean
    generated = reverse_std.reshape(SAMPLES,seq_length,-1).detach().cpu().numpy()
    return generated

#pdb.set_trace()
#print(generated)
"""
from util import dist, dataset, heatmap
hausdorff = dist.Hausdorff()
dtw = dist.DTW()
fde = dist.FDE()



print(hausdorff.update(train_array, generated))
print(dtw.update(train_array, generated))
print(fde.update(train_array, generated))
"""