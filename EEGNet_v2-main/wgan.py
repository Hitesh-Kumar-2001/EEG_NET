import argparse
import numpy as np
import random
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch

parser = argparse.ArgumentParser()
n_epochs=100
lr=0.000001
b1=0.1
b2=0.999
n_cpu=4
latent_dim=32
img_size=28
channels=1
n_critic=10
clip_value=0.01
sample_interval=200
nz=64


gen_size = 500
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

batch_size = 100
dropout_level = 0.05
nz = nz
img_shape = (9, 1500)
T = 3.0


def weights_init(m):
    if isinstance(m, nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.nz = nz+4
        self.layer1 = nn.Sequential(
            nn.Linear(self.nz, 16*26),
            nn.PReLU()
        )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=16, out_channels=18, kernel_size=20, stride=4),
            nn.PReLU()
        )
        self.layer3 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=18, out_channels=20, kernel_size=14, stride=3),
            nn.PReLU()
        )
        self.layer4 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=20, out_channels=22, kernel_size=15, stride=3),
            nn.Sigmoid()
        )
        self.embed = nn.Embedding(4,4)

    def forward(self, z,labels):
        embedding = self.embed(labels)
        embedding = embedding[:z.shape[0], :]
        z = torch.cat([z, embedding], dim=1)

        out = self.layer1(z)
        out = out.view(out.size(0), 16, 26)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=22+1, out_channels=20, kernel_size=4, stride=2),
            nn.Conv1d(in_channels=20, out_channels=16, kernel_size=3, stride=1),
            nn.LayerNorm([16,559]),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(2))
        self.dense_layers = nn.Sequential(

            nn.Linear(4464, 4000),
            nn.Linear(4000, 3000),
            nn.Linear(3000, 2000),
            nn.Linear(2000, 1000),
            nn.Linear(1000,500),
            nn.LeakyReLU(0.2),
            nn.Linear(500,200),
            nn.Linear(200, 100),
            nn.Linear(100, 1))
        self.embed = nn.Embedding(4,1125)

    def forward(self, x,labels):
        embedding = self.embed(labels.to(torch.int64))  # .veiw(labels.shape[0],1,1125)
        embedding = torch.reshape(embedding, (labels.shape[0], 1, 1125))
        embedding = embedding[:x.shape[0], :, :]
        x = torch.cat([x, embedding], dim=1)
        out = self.layer1(x)
        out = out.view(out.size(0), -1)
        out = self.dense_layers(out)
        return out


# Loss weight for gradient penalty
lambda_gp = 10

# Initialize generator and discriminator
discriminator = Discriminator()
generator = Generator()
discriminator.apply(weights_init)
generator.apply(weights_init)

discriminator.to(device)
generator.to(device)


def wgan(datatrain, label, nseed):
    random.seed(nseed)
    np.random.seed(nseed)
    torch.manual_seed(nseed)
    torch.cuda.manual_seed(nseed)

    datatrain = torch.from_numpy(datatrain)
    label = torch.from_numpy(label)


    dataset = torch.utils.data.TensorDataset(datatrain, label)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    def compute_gradient_penalty(D, real_samples, fake_samples,labels):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = Tensor(np.random.random((real_samples.size(0), 1, 1)))
        # Get random interpolation between real and fake samples
        interpolates = ((alpha * real_samples) + ((1 - alpha) * fake_samples)).requires_grad_(True)
        # print (interpolates.shape)
        d_interpolates = D(interpolates,labels)
        fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    # ----------
    #  Training
    # ----------
    new_data = []
    batches_done = 0
    discriminator.train()
    generator.train()
    for epoch in range(n_epochs):
        for i, data in enumerate(dataloader, 0):

            imgs, labels = data
            imgs = imgs.to(device)
            labels = labels.to(device)

            # Configure input
            real_data = imgs.type(Tensor)


            optimizer_D.zero_grad()
            # Sample noise as generator input
            z = torch.randn(imgs.shape[0], nz).to(device)
            # Generate a batch of images

            fake_imgs = generator(z[:imgs.shape[0],:],labels)
            # Real images
            real_validity = discriminator(real_data,labels)
            # Fake images

            fake_validity = discriminator(fake_imgs,labels)
            # Gradient penalty
            temp_loss = torch.mean(real_validity).detach()
            gradient_penalty = compute_gradient_penalty(discriminator, real_data.data, fake_imgs.data,labels)
            # Adversarial loss
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
            d_loss.backward()
            optimizer_D.step()

            # Train the generator every n_critic steps
            if i % n_critic == 0:
                optimizer_G.zero_grad()
                # -----------------
                #  Train Generator
                # -----------------
                z = torch.randn(imgs.shape[0], nz).to(device)
                # Generate a batch of images

                fake_imgs = generator(z,labels)
                # Train on fake images

                fake_validity = discriminator(fake_imgs,labels)
                g_loss = temp_loss-torch.mean(fake_validity)
                g_loss.backward()
                optimizer_G.step()

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
                )

    discriminator.eval()
    generator.eval()
    data = []
    label = []
    for i in range(0, 4):
        z = torch.randn(gen_size, nz).to(device)
        labels = [i for j in range(gen_size)]
        labels = torch.tensor(labels).to(device)
        data.append(generator(z, labels).cpu().detach().numpy())
        label.append(labels.cpu().detach().numpy())

    return np.array(data), np.array(label)
