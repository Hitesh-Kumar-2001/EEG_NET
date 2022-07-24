import torch
import torch.nn as nn
import numpy as np
import os
import random
import argparse

parser = argparse.ArgumentParser()
n_epochs=150
lr=0.00001
b1=0.5
b2=0.999
n_cpu=8
latent_dim=320
img_size=32
n_critic=10
channels=1
sample_interval=200
nz=64
gen_size = 500

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

batch_size = 100
learning_rate = 0.00001
dropout_level = 0.05
nz = nz


def weights_init(m):
    if isinstance(m, nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

class EEG_CNN_Generator(nn.Module):
    def __init__(self):
        super(EEG_CNN_Generator, self).__init__()

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
        self.embed = nn.Embedding(4,16)

    def forward(self, z,labels):
        labels = labels.repeat(4, 1)
        labels = torch.transpose(labels, 0, 1)


        z = torch.cat([z, labels], dim=1)

        out = self.layer1(z)
        out = out.view(out.size(0), 16, 26)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        return out

class EEG_CNN_Discriminator(nn.Module):
    def __init__(self):
        super(EEG_CNN_Discriminator, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=22+1, out_channels=16, kernel_size=10, stride=2),
            nn.BatchNorm1d(num_features=16),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(2))
        self.dense_layers = nn.Sequential(

            nn.Linear(4464, 4000),
            nn.Linear(4000, 3000),
            nn.Linear(3000, 2000),
            nn.Linear(2000, 1000),
            nn.Linear(1000, 500),
            nn.LeakyReLU(0.2),
            nn.Linear(500, 200),
            nn.Linear(200, 100),
            nn.Linear(100, 1))
        self.embed = nn.Embedding(4, 1125)

    def forward(self, x,labels):
        labels = labels.repeat(1125, 1)
        labels = torch.transpose(labels, 0, 1)
        labels = torch.reshape(labels, (labels.shape[0], 1, -1))
        x = torch.cat([x, labels], dim=1)
        out = self.layer1(x)
        out = out.view(out.size(0), -1)
        out = self.dense_layers(out)
        return out

def dcgan(datatrain, label, nseed):
    random.seed(nseed)
    np.random.seed(nseed)
    torch.manual_seed(nseed)
    torch.cuda.manual_seed(nseed)

    datatrain = torch.from_numpy(datatrain)
    label = torch.from_numpy(label)

    dataset = torch.utils.data.TensorDataset(datatrain, label)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    generator = EEG_CNN_Generator().to(device)
    discriminator = EEG_CNN_Discriminator().to(device)
    discriminator.apply(weights_init)
    generator.apply(weights_init)

    # Loss function
    adversarial_loss = nn.BCEWithLogitsLoss()

    # Optimizer
    optimizer_Gen = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(b1, b2))
    optimizer_Dis = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    real_label = 1
    fake_label = 0
    batches_done = 0
    new_data = []
    global_step = 0

    # GAN Training ---------------------------------------------------------------
    discriminator.train()
    generator.train()
    for epoch in range(n_epochs):
        for i, data in enumerate(dataloader, 0):
            imgs, labels = data
            imgs = imgs.to(device)
            labels = labels.to(device)

            # Configure input
            real_data = imgs.type(Tensor)
            labels = labels.type(Tensor)
            label = torch.ones(imgs.shape[0], 1).to(device)
            z = torch.randn(imgs.shape[0], nz).to(device)

            if (epoch % 1) == 0:

                optimizer_Dis.zero_grad()
                output_real = discriminator(real_data,labels)

                # Calculate error and backpropagate
                errD_real = adversarial_loss(output_real, label)
                errD_real.backward()

                fake_labels = torch.randint(0, 4, (imgs.shape[0],)).to(device)
                fake_data = generator(z,fake_labels)
                label = torch.zeros(imgs.shape[0], 1).to(device)
                output_fake = discriminator(fake_data,fake_labels)
                errD_fake = adversarial_loss(output_fake, label)
                errD_fake.backward()
                errD = errD_real + errD_fake
                optimizer_Dis.step()


            if i % 1 == 0:
                fake_labels = torch.randint(0, 4, (imgs.shape[0],)).to(device)
                z = torch.randn(imgs.shape[0], nz).to(device)
                fake_data = generator(z,fake_labels)

                # Reset gradients
                optimizer_Gen.zero_grad()

                output = discriminator(fake_data,fake_labels)
                bceloss = adversarial_loss(output, torch.zeros(imgs.shape[0], 1).to(device))
                errG = bceloss
                errG.backward()
                optimizer_Gen.step()

            print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] " % (
            epoch, n_epochs, i, len(dataloader), errD.item(), errG.item(),))

    # generate the data
    discriminator.eval()
    generator.eval()
    data = []
    label = []
    for i in range(0,4):
        z = torch.randn(gen_size, nz).to(device)
        labels = [i for j in range(gen_size)]
        labels = torch.tensor(labels).to(device)
        data.append(generator(z,labels).cpu().detach().numpy())
        label.append(labels.cpu().detach().numpy())

    return np.array(data),np.array(label)









#####################################################

