import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# hyperparameters
num_epochs = 300
batch_size = 100


def weights_init(m):
    if isinstance(m, nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size=16):
        UF = input.view(input.size(0), size, 26)
        return UF


class EEG_CNN_VAE(nn.Module):
    def __init__(self):
        super(EEG_CNN_VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=22+1, out_channels=16, kernel_size=10, stride=2),
            nn.BatchNorm1d(num_features=16),
            nn.PReLU(),
            nn.MaxPool1d(2),
            Flatten()
        )

        self.fc1 = nn.Linear(4464, 16)
        self.fc2 = nn.Linear(4464, 16)
        self.fc3 = nn.Linear(16+4, 16*26)

        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose1d(in_channels=16, out_channels=18, kernel_size=20, stride=4),
            nn.PReLU(),
            nn.ConvTranspose1d(in_channels=18, out_channels=20, kernel_size=14, stride=3),
            nn.PReLU(),
            nn.ConvTranspose1d(in_channels=20, out_channels=22, kernel_size=15, stride=3),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_().to(device)
        esp = torch.randn(*mu.size()).to(device)
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self,x,labels):

        labels = labels.repeat(1125,1)
        labels = torch.transpose(labels,0,1)
        labels = torch.reshape(labels,(labels.shape[0],1,-1))
        x = torch.cat([x,labels],dim=1)
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)

        return z, mu, logvar

    def decode(self, z,labels):

        labels = labels.repeat(4,1)
        labels = torch.transpose(labels, 0, 1)

        z = torch.cat([z, labels], dim=1)


        z = self.fc3(z)
        z = self.decoder(z)

        return z

    def forward(self, x,labels):

        z, mu, logvar = self.encode(x,labels)
        z = self.decode(z,labels)
        return z, mu, logvar


def loss_fn(recon_x, x, mu, logvar):

    BCE = F.binary_cross_entropy(recon_x, x)
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return (BCE + KLD)/x.size(0), BCE, KLD


def vae(datatrain, label, nseed):
    random.seed(nseed)
    np.random.seed(nseed)
    torch.manual_seed(nseed)
    torch.cuda.manual_seed(nseed)

    datatrain = torch.from_numpy(datatrain)
    label = torch.from_numpy(label)

    dataset = torch.utils.data.TensorDataset(datatrain, label)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    model = EEG_CNN_VAE().to(device)
    model.apply(weights_init)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        for i, data_train in enumerate(dataloader, 0):
            data, labels = data_train
            data = data.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            recon_data, mu, logvar = model(data,labels)
            loss, bce, kld = loss_fn(recon_data, data, mu, logvar)

            loss.backward()
            optimizer.step()

            to_print = "Epoch[{}/{}] Loss: {:.6f} {:.6f} {:.6f}".format(epoch+1, num_epochs, loss.item(), bce.item(), kld.item())
            print(to_print)

    # Generating new data
    new_data = []
    labels = []
    num_data_to_generate = 500
    with torch.no_grad():
        model.eval()
        for i in range(4):

            for epoch in range(num_data_to_generate):
                z = torch.randn(1, 16).to(device)
                recon_data = model.decode(z,torch.Tensor([i]).to(device)).cpu().numpy()
                labels.append(i)
                new_data.append(recon_data)


        # new_data = np.concatenate(new_data)
        new_data = np.asarray(new_data)
        labels = np.asarray(labels)
        return new_data, labels

