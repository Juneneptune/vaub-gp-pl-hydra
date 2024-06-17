import torch
import torch.nn as nn
import torch.nn.functional as F

# Define a Residual Block
class ResidualBlockVAE(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlockVAE, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

# Define the Encoder with Skip Connections and Residual Blocks
class Encoder(nn.Module):
    def __init__(self, latent_size=64, is_svhn=False):
        super(Encoder, self).__init__()
        self.latent_size = latent_size
        self.conv1 = nn.Conv2d(1, 16, 4, 2, 1)  # (batch_size, 32, 14, 14)
        self.res1 = ResidualBlockVAE(16, 16)
        self.conv2 = nn.Conv2d(16, 64, 4, 2, 1)  # (batch_size, 64, 7, 7)
        self.res2 = ResidualBlockVAE(64, 64)
        if is_svhn:
            self.conv3 = nn.Conv2d(64, 2*latent_size, 4, 2, 1)
        else:
            self.conv3 = nn.Conv2d(64, 2*latent_size, 3, 2, 1)  # (batch_size, 128, 4, 4)
        self.res3 = ResidualBlockVAE(2*latent_size, 2*latent_size)
        self.conv4 = nn.Conv2d(2*latent_size, 2*latent_size, 4)  # (batch_size, 128, 1, 1)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.res1(x)
        x = F.relu(self.conv2(x))
        x = self.res2(x)
        x = F.relu(self.conv3(x))
        x = self.res3(x)
        
        mu, logvar = self.conv4(x).view(x.shape[0], -1).chunk(2, dim=-1)  # (batch_size, latent_size)
        
        mu = mu.view((-1, self.latent_size, 1, 1))
        logvar = logvar.view((-1, self.latent_size, 1, 1))
        
        return mu, logvar

# Define the Decoder with Skip Connections and Residual Blocks
class Decoder(nn.Module):
    def __init__(self, latent_size=64, is_svhn=False):
        super(Decoder, self).__init__()
        self.latent_size = latent_size
        self.res1 = ResidualBlockVAE(latent_size, latent_size)
        self.conv1 = nn.ConvTranspose2d(latent_size, 64, 4)  # (batch_size, 64, 4, 4)
        self.res2 = ResidualBlockVAE(64, 64)
        self.conv2 = nn.ConvTranspose2d(64, 16, 4, 2, 1)  # (batch_size, 32, 8, 8)
        self.res3 = ResidualBlockVAE(16, 16)
        if is_svhn:
            self.conv3 = nn.ConvTranspose2d(16, 1, 4, 4, 2)
        else:
            self.conv3 = nn.ConvTranspose2d(16, 1, 4, 4, 2) 

    def forward(self, x):
        x = self.res1(x)
        x = F.relu(self.conv1(x))
        x = self.res2(x)
        x = F.relu(self.conv2(x))
        x = self.res3(x)
        x = torch.sigmoid(self.conv3(x))  # Sigmoid activation for binary image
        return x

# Define the VAE with Skip Connections and Residual Blocks
class CNN_VAE(nn.Module):
    def __init__(self, latent_height=8, latent_width=8, is_svhn=False):
        super(VAE, self).__init__()
        self.latent_height = latent_height
        self.latent_width = latent_width
        
        self.encoder = Encoder(latent_size=latent_height*latent_width, is_svhn=is_svhn)
        self.decoder = Decoder(latent_size=latent_height*latent_width, is_svhn=is_svhn)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z.view((z.shape[0], -1, 1, 1)))

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), z.view((z.shape[0], 1, self.latent_height, self.latent_width)), mu.view((z.shape[0], -1)), logvar.view((z.shape[0], -1))


# class Encoder(nn.Module):
#     def __init__(self):
#         super(Encoder, self).__init__()
#         self.conv1 = nn.Conv2d(1, 4, kernel_size=4, stride=2, padding=1)  # Output: 32x14x14
#         self.conv2 = nn.Conv2d(4, 16, kernel_size=4, stride=2, padding=1)  # Output: 64x7x7
#         self.conv3 = nn.Conv2d(16, 64, kernel_size=3, stride=2, padding=1)  # Output: 128x4x4
#         self.fc1 = nn.Linear(64 * 4 * 4, 256)
#         self.fc2_mu = nn.Linear(256, 1 * 16 * 16)
#         self.fc2_logvar = nn.Linear(256, 1 * 16 * 16)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv3(x))
#         x = x.view(x.size(0), -1)
#         x = F.relu(self.fc1(x))
#         mu = self.fc2_mu(x)
#         logvar = self.fc2_logvar(x)
#         return mu, logvar


# class Decoder(nn.Module):
#     def __init__(self):
#         super(Decoder, self).__init__()
#         self.fc = nn.Linear(256, 64 * 7 * 7)  # Adjusted to match the size before reshaping
#         self.deconv1 = nn.ConvTranspose2d(64, 16, kernel_size=4, stride=2, padding=1)
#         self.deconv2 = nn.ConvTranspose2d(16, 4, kernel_size=4, stride=2, padding=1)
#         self.deconv3 = nn.ConvTranspose2d(4, 1, kernel_size=4, stride=2, padding=1)
#         self.fc1 = nn.Linear(56 * 56, 28 * 28)

#     def forward(self, x):
#         x = F.relu(self.fc(x))
#         x = x.view(x.size(0), 64, 7, 7)  # Reshape to match the size before upsampling
#         x = F.relu(self.deconv1(x))
#         x = F.relu(self.deconv2(x))
#         x = F.relu(self.deconv3(x)).view((x.shape[0], -1))
#         x = torch.sigmoid(self.fc1(x))
#         return x


# class CNN_VAE(nn.Module):
#     def __init__(self):
#         super(CNN_VAE, self).__init__()
#         self.encoder = Encoder()
#         self.decoder = Decoder()

#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mu + eps * std

#     def decode(self, z):
#         return self.decoder(z)

#     def forward(self, x):
#         mu, logvar = self.encoder(x)
#         z = self.reparameterize(mu, logvar)
#         recon = self.decoder(z)
#         # print(f'recon: {recon.shape}')
#         return recon.view((z.shape[0], -1)), z.view((z.shape[0], -1)), mu.view((z.shape[0], -1)), logvar.view(
#             (z.shape[0], -1))


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.Sigmoid(),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim * 2)  # Mean and log-variance
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.Sigmoid(),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        mean, logvar = h.chunk(2, dim=-1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        return self.decode(z), z, mean, logvar
