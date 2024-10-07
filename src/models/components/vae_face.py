import torch
import torch.nn as nn
import torch.nn.functional as F

# Define a Residual Block for CelebA VAE
class ResidualBlockVAE(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
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

# Define the Encoder for CelebA VAE with Skip Connections and Residual Blocks
class ResConvEncoder(nn.Module):
    def __init__(self, channel_size=128, logvar_constraint='sigmoid'):
        super().__init__()
        self.channel_size = channel_size
        self.conv1 = nn.Conv2d(3, 64, 4, 2, 1)  # 64x64 -> 32x32
        self.res1 = ResidualBlockVAE(64, 64)
        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1)  # 32x32 -> 16x16
        self.res2 = ResidualBlockVAE(128, 128)
        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1)  # 16x16 -> 8x8
        self.res3 = ResidualBlockVAE(256, 256)
        self.conv4 = nn.Conv2d(256, 2 * channel_size, 4, 2, 1)  # 8x8 -> 4x4
        self.res4 = ResidualBlockVAE(2 * channel_size, 2 * channel_size)
        self.bn_mu = nn.BatchNorm2d(channel_size, affine=False)
        self.logvar_constraint = logvar_constraint

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.res1(x)
        x = F.relu(self.conv2(x))
        x = self.res2(x)
        x = F.relu(self.conv3(x))
        x = self.res3(x)
        x = self.conv4(x)

        # Reshape for mu and logvar (output will be [batch_size, 2*channel_size, 4, 4])
        mu, logvar = x.chunk(2, dim=1)  # Split into mu and log variance
        mu = self.bn_mu(mu)
        if self.logvar_constraint == 'sigmoid':
                logvar = 7 * (torch.sigmoid(logvar) - 0.5)
        elif self.logvar_constraint == 'clamp':
            logvar = torch.clamp(logvar, max=4)
        else:
            raise ValueError(f"Invalid logvar_constraint: {self.logvar_constraint}")
        return mu, logvar

# Define the Decoder for CelebA VAE with Skip Connections and Residual Blocks
class ResConvDecoder(nn.Module):
    def __init__(self, channel_size=128):
        super().__init__()
        self.res1 = ResidualBlockVAE(channel_size, channel_size)
        self.conv1 = nn.ConvTranspose2d(channel_size, 256, 4, 2, 1)  # 4x4 -> 8x8
        self.res2 = ResidualBlockVAE(256, 256)
        self.conv2 = nn.ConvTranspose2d(256, 128, 4, 2, 1)  # 8x8 -> 16x16
        self.res3 = ResidualBlockVAE(128, 128)
        self.conv3 = nn.ConvTranspose2d(128, 64, 4, 2, 1)  # 16x16 -> 32x32
        self.res4 = ResidualBlockVAE(64, 64)
        self.conv4 = nn.ConvTranspose2d(64, 3, 4, 2, 1)  # 32x32 -> 64x64

    def forward(self, x):
        x = x.view(x.size(0), -1, 4, 4)  # Reshape channel vector into 4x4 feature map
        x = self.res1(x)
        x = F.relu(self.conv1(x))
        x = self.res2(x)
        x = F.relu(self.conv2(x))
        x = self.res3(x)
        x = F.relu(self.conv3(x))
        x = self.res4(x)
        x = torch.sigmoid(self.conv4(x))  # Tanh for output pixel values between -1 and 1
        return x

# Define the VAE for CelebA dataset with Skip Connections and Residual Blocks
class ResConvVAE(nn.Module):
    def __init__(self, channel_size=16, logvar_constraint='sigmoid'):
        super().__init__()
        self.encoder = ResConvEncoder(channel_size=channel_size, logvar_constraint=logvar_constraint)
        self.decoder = ResConvDecoder(channel_size=channel_size)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, z, mu, logvar
        # return self.decoder(z), z, mu.view((z.shape[0], -1)), logvar.view((z.shape[0], -1))

    def decode(self, z):
        return self.decoder(z)

    # Weight initialization with fixed random values
    def init_weights_fixed(self, init_scale=0.1, seed=42):
        torch.manual_seed(seed)  # Set fixed seed for reproducibility

        def weights_init(m):
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.uniform_(m.weight, a=-init_scale, b=init_scale)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.apply(weights_init)
