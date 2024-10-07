import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.ln1 = nn.LayerNorm(out_dim, elementwise_affine=True)
        self.swish = nn.SiLU()
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.ln2 = nn.LayerNorm(out_dim, elementwise_affine=True)

    def forward(self, x):
        identity = x
        out = self.fc1(x)
        out = self.ln1(out)
        out = self.swish(out)
        out = self.fc2(out)
        out = self.ln2(out)
        out += identity  # Skip connection
        return self.swish(out)


# Conditional LayerNorm module for linear input
class ConditionalLayerNorm(nn.Module):
    def __init__(self, num_features, embedding_dim):
        super().__init__()
        # LayerNorm without affine parameters for normalization over linear features
        self.layer_norm = nn.LayerNorm(num_features, elementwise_affine=False)
        
        # Linear layer to conditionally compute gamma (scale) and beta (shift)
        self.embed = nn.Linear(embedding_dim, num_features * 2, bias=False)

        # No need to initialize gamma and beta here, let the linear layer handle it.

    def forward(self, x, cond):
        # Apply layer normalization (without learned affine parameters)
        x = self.layer_norm(x)

        # Compute gamma and beta from the conditioning embedding
        gamma, beta = self.embed(cond).chunk(2, dim=1)  # Split into two parts

        # No need to reshape as the input is 1D (batch_size, num_features)
        # Apply conditional scaling (gamma) and shifting (beta)
        x = x * gamma + beta
        
        return x

# Residual Block with Conditional Group Norm
class ConditionalResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, embedding_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.norm1 = ConditionalLayerNorm(num_features=out_dim, embedding_dim=embedding_dim)
        self.swish = nn.SiLU()
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.norm2 = ConditionalLayerNorm(num_features=out_dim, embedding_dim=embedding_dim)

    def forward(self, x, cond):
        identity = x
        out = self.fc1(x)
        out = self.norm1(out, cond)
        out = self.swish(out)
        out = self.fc2(out)
        out = self.norm2(out, cond)
        out += identity  # Skip connection
        return self.swish(out)

# Conditional Group Normalization for 2D inputs
class CNNConditionalGroupNorm2D(nn.Module):
    def __init__(self, num_groups, num_channels, embedding_dim):
        super().__init__()
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels, affine=False)
        self.embed = nn.Linear(embedding_dim, num_channels * 2)

        self.embed.weight.data[:, :num_channels] = 1.0  # gamma initialization
        self.embed.weight.data[:, num_channels:] = 0.0  # beta initialization

    def forward(self, x, cond):
        x = self.group_norm(x)
        gamma, beta = self.embed(cond).chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return x * gamma + beta

# Residual Block with Conditional Group Norm for CNNs
class CNNResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, embedding_dim):
        super().__init__()

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embedding_dim, out_channels)
        )

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = CNNConditionalGroupNorm2D(num_groups=32, num_channels=out_channels, embedding_dim=embedding_dim)
        self.swish = nn.SiLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = CNNConditionalGroupNorm2D(num_groups=32, num_channels=out_channels, embedding_dim=embedding_dim)

        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        identity = self.residual_conv(x)
        time_emb = self.time_mlp(cond)[:, :, None, None]

        out = self.conv1(x)
        out = self.norm1(out, cond)
        out = self.swish(out + time_emb)

        out = self.conv2(out)
        out = self.norm2(out, cond)
        out += identity
        return self.swish(out)

# Self-Attention block adapted for CNNs (Channel Attention)
class CNNSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv_q = nn.Conv2d(in_channels, in_channels, 1)
        self.conv_k = nn.Conv2d(in_channels, in_channels, 1)
        self.conv_v = nn.Conv2d(in_channels, in_channels, 1)
        self.scale = torch.sqrt(torch.FloatTensor([in_channels])).cuda()

    def forward(self, x):
        B, C, H, W = x.shape
        Q = self.conv_q(x).view(B, C, H * W)
        K = self.conv_k(x).view(B, C, H * W)
        V = self.conv_v(x).view(B, C, H * W)

        attention = torch.softmax(Q @ K.transpose(-2, -1) / self.scale, dim=-1)
        out = attention @ V
        out = out.view(B, C, H, W)
        return out

class CNNCrossAttention(nn.Module):
    def __init__(self, in_channels, embedding_dim):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key = nn.Linear(embedding_dim, in_channels)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.scale = in_channels ** -0.5

    def forward(self, x, cond):
        B, C, H, W = x.shape  # Batch size, channels, height, width

        # Prepare the query, key, and value matrices
        Q = self.query(x).view(B, C, H * W).permute(0, 2, 1)  # (B, H*W, C)
        K = self.key(cond).unsqueeze(1)  # (B, 1, C)
        V = self.value(x).view(B, C, H * W).permute(0, 2, 1)  # (B, H*W, C)

        # Perform the attention mechanism
        attention = torch.softmax(Q @ K.transpose(-2, -1) * self.scale, dim=1)  # (B, H*W, 1)

        # Multiply the attention by the value
        out = attention * V  # (B, H*W, C)

        # Reshape back to original spatial dimensions
        out = out.permute(0, 2, 1).view(B, C, H, W)  # (B, C, H, W)
        return out


# UNet with advanced techniques
class UNet(nn.Module):
    def __init__(self, in_dim, out_dim, num_timesteps, embedding_dim=2, multiplier=4, is_warm_init=False):
        super(UNet, self).__init__()
        self.num_timesteps = num_timesteps

        # Define encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, multiplier*in_dim),
            nn.SiLU(),
            ResidualBlock(multiplier*in_dim, multiplier*in_dim),
            nn.Dropout(0.1),
            ResidualBlock(multiplier*in_dim, multiplier*in_dim),
            nn.Dropout(0.1)
        )

        # Define decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(multiplier*in_dim + embedding_dim, multiplier*in_dim),
            nn.SiLU(),
            ResidualBlock(multiplier*in_dim, multiplier*in_dim),
            nn.Dropout(0.1),
            ResidualBlock(multiplier*in_dim, multiplier*in_dim),
            nn.Dropout(0.1),
            nn.Linear(multiplier*in_dim, out_dim)
        )

        # Define time step embedding layer for decoder
        self.embedding = nn.Embedding(num_timesteps, embedding_dim)

        if is_warm_init:
            self.warm_init()

    def forward(self, x, timestep, enc_sigma=None):
        # Encoder
        if enc_sigma is not None:
            encoded_enc_sigma = self.encoder(enc_sigma)
        else:
            encoded_enc_sigma = 0
        x = self.encoder(x) + encoded_enc_sigma

        # Decoder
        x = self.decoder(torch.hstack((x, self.embedding(timestep))))

        return x

    def warm_init(self):
        # Custom initialization for better convergence
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.01)


class UNet_noise(nn.Module):
    def __init__(self, in_dim, out_dim, num_timesteps, embedding_dim=2, num_latent_noise_scale=50, is_add_latent_noise=False, multiplier=4, is_warm_init=False):
        super(UNet_noise, self).__init__()
        self.num_timesteps = num_timesteps
        self.is_add_latent_noise = is_add_latent_noise

        # Define encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, multiplier*in_dim),
            nn.SiLU(),
            ResidualBlock(multiplier*in_dim, multiplier*in_dim),
            # nn.Dropout(0.1),
            ResidualBlock(multiplier*in_dim, multiplier*in_dim),
            # nn.Dropout(0.1)
        )

        # Define decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(multiplier*in_dim + embedding_dim, multiplier*in_dim) if not is_add_latent_noise else nn.Linear(multiplier*in_dim + embedding_dim + embedding_dim, multiplier*in_dim),
            nn.SiLU(),
            ResidualBlock(multiplier*in_dim, multiplier*in_dim),
            # nn.Dropout(0.1),
            ResidualBlock(multiplier*in_dim, multiplier*in_dim),
            # nn.Dropout(0.1),
            nn.Linear(multiplier*in_dim, out_dim)
        )

        # Define time step embedding layer for decoder
        self.embedding = nn.Embedding(num_timesteps, embedding_dim)
        if self.is_add_latent_noise:
            self.latent_noise_embedding = nn.Embedding(num_latent_noise_scale, embedding_dim)
        if is_warm_init:
            self.warm_init()

    def forward(self, x, timestep, latent_noise_idx=None, enc_sigma=None):
        # Encoder
        if enc_sigma is not None:
            encoded_enc_sigma = self.encoder(enc_sigma)
        else:
            encoded_enc_sigma = 0
        x = self.encoder(x) + encoded_enc_sigma

        # Decoder
        if self.is_add_latent_noise:
            if latent_noise_idx is None:
                latent_noise_idx = torch.zeros(x.shape[0], device=x.device).type(torch.long)
            x = self.decoder(torch.hstack((x, self.embedding(timestep), self.latent_noise_embedding(latent_noise_idx))))
        else:
            x = self.decoder(torch.hstack((x, self.embedding(timestep))))

        return x

    def warm_init(self):
        # Custom initialization for better convergence
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.01)


# UNet for Score-Based Model with CNN layers and Cross-Attention
class CNNAdvancedUNet(nn.Module):
    def __init__(self, in_channels, num_timesteps, ch_multi=2, embedding_dim=128, base_channels=64, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        super().__init__()
        self.device = device

        # Time embedding
        self.time_embedding = nn.Sequential(
            SinusoidalPosEmb(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

        # Encoder with residual blocks
        self.encoder1 = CNNResidualBlock(in_channels, base_channels, embedding_dim)
        self.encoder2 = CNNResidualBlock(base_channels, base_channels * ch_multi, embedding_dim)
        self.encoder3 = CNNResidualBlock(base_channels * ch_multi, base_channels * ch_multi ** 2, embedding_dim)

        # Attention block
        self.attention = CNNSelfAttention(base_channels * ch_multi ** 2)

        # Cross-attention block between feature maps and time embedding
        self.cross_attention = CNNCrossAttention(base_channels * ch_multi ** 2, embedding_dim)

        # Decoder
        self.decoder1 = CNNResidualBlock(base_channels * ch_multi ** 2, base_channels * ch_multi, embedding_dim)
        self.decoder2 = CNNResidualBlock(base_channels * ch_multi, base_channels, embedding_dim)
        self.decoder3 = nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x, timestep):
        # Time embedding
        t_emb = self.time_embedding(timestep)

        # Encoder path
        x = self.encoder1(x, t_emb)
        x = self.encoder2(x, t_emb)
        x = self.encoder3(x, t_emb)

        # Attention
        x = self.attention(x)

        # Cross-attention
        x = self.cross_attention(x, t_emb)

        # Decoder path
        x = self.decoder1(x, t_emb)
        x = self.decoder2(x, t_emb)
        x = self.decoder3(x)
        return x


# UNet with more advanced techniques
class AdvancedUNet(nn.Module):
    def __init__(self, in_dim, out_dim, num_timesteps, embedding_dim=4, num_latent_noise_scale=50, is_add_latent_noise=False, multiplier=4, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), is_warm_init=False, *args, **kwargs):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.device = device
        self.is_add_latent_noise = is_add_latent_noise

        # Time embedding
        self.time_embedding = nn.Sequential(
            SinusoidalPosEmb(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

        # Encoder
        self.LinEnc1 = nn.Linear(in_dim, multiplier * in_dim)
        self.silu = nn.SiLU()
        self.ResEnc1 = ConditionalResidualBlock(multiplier * in_dim, multiplier * in_dim, embedding_dim)
        self.ResEnc2 = ConditionalResidualBlock(multiplier * in_dim, multiplier * in_dim, embedding_dim)
        self.ResEnc3 = ConditionalResidualBlock(multiplier * in_dim, multiplier * in_dim, embedding_dim)
        
        # Decoder
        self.ResDec1 = ConditionalResidualBlock(multiplier * in_dim, multiplier * in_dim, embedding_dim)
        self.ResDec2 = ConditionalResidualBlock(multiplier * in_dim, multiplier * in_dim, embedding_dim)
        self.ResDec3 = ConditionalResidualBlock(multiplier * in_dim, multiplier * in_dim, embedding_dim)
        self.LinDec1 = nn.Linear(multiplier * in_dim, out_dim)


        if is_warm_init:
            self.warm_init()

    def forward(self, x, timestep, latent_noise_idx=None, enc_sigma=None):

        # Cross attention between encoder output and time embedding
        t_emb = self.time_embedding(timestep)
        x = self.LinEnc1(x)
        x = self.silu(x)
        x = self.ResEnc1(x, t_emb)
        x = self.ResEnc2(x, t_emb)
        x = self.ResEnc3(x, t_emb)

        # Decode
        x = self.ResDec1(x, t_emb)
        x = self.ResDec2(x, t_emb)
        x = self.ResDec3(x, t_emb)
        x = self.LinDec1(x)
        return x


# Sinusoidal Positional Embedding for Timesteps
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = torch.exp(torch.arange(half_dim, device=device) * -(torch.log(torch.tensor(10000.0)) / (half_dim - 1)))
        emb = x[:, None] * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)