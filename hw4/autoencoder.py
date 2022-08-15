import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        modules = []

        # TODO:
        #  Implement a CNN. Save the layers in the modules list.
        #  The input shape is an image batch: (N, in_channels, H_in, W_in).
        #  The output shape should be (N, out_channels, H_out, W_out).
        #  You can assume H_in, W_in >= 64.
        #  Architecture is up to you, but it's recommended to use at
        #  least 3 conv layers. You can use any Conv layer parameters,
        #  use pooling or only strides, use any activation functions,
        #  use BN or Dropout, etc.
        # ====== YOUR CODE: ======
        channels = [64,256,512,out_channels]
        kernel_size = 3
        stride = 2
        padding = 1
        for out_channel in channels:
            conv_layer = nn.Conv2d(in_channels,out_channel,kernel_size,stride,padding)
            norm_layer = nn.BatchNorm2d(out_channel)
            activation_layer = nn.ReLU()

            modules.append(conv_layer)
            modules.append(norm_layer)
            modules.append(activation_layer)

            in_channels = out_channel
        # ========================
        self.cnn = nn.Sequential(*modules)

    def forward(self, x):
        return self.cnn(x)


class DecoderCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        modules = []

        # TODO:
        #  Implement the "mirror" CNN of the encoder.
        #  For example, instead of Conv layers use transposed convolutions,
        #  instead of pooling do unpooling (if relevant) and so on.
        #  The architecture does not have to exactly mirror the encoder
        #  (although you can), however the important thing is that the
        #  output should be a batch of images, with same dimensions as the
        #  inputs to the Encoder were.
        # ====== YOUR CODE: ======
        channels = [in_channels,512,256,64,out_channels]
        kernel_size = 3
        stride = 2
        padding = 1
        for i in range(len(channels) -1):
            conv_layer = nn.ConvTranspose2d(channels[i], channels[i+1], kernel_size, stride, padding,padding)
            norm_layer = nn.BatchNorm2d(channels[i+1])
            activation_layer = nn.ReLU()

            modules.append(conv_layer)
            modules.append(norm_layer)
            modules.append(activation_layer)
        # ========================
        self.cnn = nn.Sequential(*modules)

    def forward(self, h):
        # Tanh to scale to [-1, 1] (same dynamic range as original images).
        return torch.tanh(self.cnn(h))


class VAE(nn.Module):
    def __init__(self, features_encoder, features_decoder, in_size, z_dim):
        """
        :param features_encoder: Instance of an encoder the extracts features
        from an input.
        :param features_decoder: Instance of a decoder that reconstructs an
        input from it's features.
        :param in_size: The size of one input (without batch dimension).
        :param z_dim: The latent space dimension.
        """
        super().__init__()
        self.features_encoder = features_encoder
        self.features_decoder = features_decoder
        self.z_dim = z_dim

        self.features_shape, n_features = self._check_features(in_size)

        # TODO: Add more layers as needed for encode() and decode().
        # ====== YOUR CODE: ======
        self.mean = torch.nn.Linear(16384,self.z_dim)
        self.log_var = torch.nn.Linear(16384, self.z_dim)

        self.latent_to_decoder = torch.nn.Linear(z_dim, 16384)
        # ========================

    def _check_features(self, in_size):
        device = next(self.parameters()).device
        with torch.no_grad():
            # Make sure encoder and decoder are compatible
            x = torch.randn(1, *in_size, device=device)
            h = self.features_encoder(x)
            xr = self.features_decoder(h)
            assert xr.shape == x.shape
            # Return the shape and number of encoded features
            return h.shape[1:], torch.numel(h) // h.shape[0]

    def encode(self, x):
        # TODO:
        #  Sample a latent vector z given an input x from the posterior q(Z|x).
        #  1. Use the features extracted from the input to obtain mu and
        #     log_sigma2 (mean and log variance) of q(Z|x).
        #  2. Apply the reparametrization trick to obtain z.
        # ====== YOUR CODE: ======
        # Extract image features
        h = self.features_encoder(x)
        h = torch.flatten(h,start_dim=1)
        # Transform into mu and log_sigma2
        mu = self.mean(h)
        log_sigma2 = self.log_var(h)
        # Reparametrization trick: Sample u from an isotropic gaussian and
        # use it to create z.
        u = torch.rand_like(log_sigma2)
        z = mu + u * log_sigma2
        # ========================
        return z, mu, log_sigma2

    def decode(self, z):
        # TODO:
        #  Convert a latent vector back into a reconstructed input.
        #  1. Convert latent z to features h with a linear layer.
        #  2. Apply features decoder.
        # ====== YOUR CODE: ======
        h_bar = self.latent_to_decoder(z)

        h_bar = h_bar.view(-1, 1024, 4, 4)

        x_rec = self.features_decoder(h_bar)
        # ========================
        # Scale to [-1, 1] (same dynamic range as original images).
        return torch.tanh(x_rec)

    def sample(self, n):
        samples = []
        device = next(self.parameters()).device
        with torch.no_grad():
            # TODO:
            #  Sample from the model. Generate n latent space samples and
            #  return their reconstructions.
            #  Notes:
            #  - Remember that this means using the model for INFERENCE.
            #  - We'll ignore the sigma2 parameter here:
            #    Instead of sampling from N(psi(z), sigma2 I), we'll just take
            #    the mean, i.e. psi(z).
            # ====== YOUR CODE: ======
            z = torch.randn(size=(n,self.z_dim))
            z = z.to(device)
            samples = self.decode(z)
            # ========================
        # Detach and move to CPU for display purposes
        samples = [s.detach().cpu() for s in samples]
        return samples

    def forward(self, x):
        z, mu, log_sigma2 = self.encode(x)
        return self.decode(z), mu, log_sigma2


def vae_loss(x, xr, z_mu, z_log_sigma2, x_sigma2):
    """
    Point-wise loss function of a VAE with latent space of dimension z_dim.
    :param x: Input image batch of shape (N,C,H,W).
    :param xr: Reconstructed (output) image batch.
    :param z_mu: Posterior mean (batch) of shape (N, z_dim).
    :param z_log_sigma2: Posterior log-variance (batch) of shape (N, z_dim).
    :param x_sigma2: Likelihood variance (scalar).
    :return:
        - The VAE loss
        - The data loss term
        - The KL divergence loss term
    all three are scalars, averaged over the batch dimension.
    """
    loss, data_loss, kldiv_loss = None, None, None
    # TODO:
    #  Implement the VAE pointwise loss calculation.
    #  Remember:
    #  1. The covariance matrix of the posterior is diagonal.
    #  2. You need to average over the batch dimension.
    # ====== YOUR CODE: ======
    z_dim = z_mu.shape[1]
    x_dim = x.shape[0]

    scaler = 1 / (x_sigma2 * x_dim)

    #data_loss = scaler * torch.linalg.norm(x-xr)

    data_loss_sum = 0.0
    for i in range(x_dim):
        data_loss_sum += scaler * (torch.linalg.norm(x[i] - xr[i]))
    data_loss = data_loss_sum / x_dim

    #print("data_loss:",data_loss)

    kl_sum = 0.0

    for i in range(x_dim):
        kl_sum += torch.sum(z_log_sigma2[i] ** 2,dim=0)  # the trace part
        kl_sum += torch.linalg.norm(z_mu[i]) ** 2
        kl_sum -= z_dim
        kl_sum -= torch.sum(z_log_sigma2[i], dim=0)  # the determinant of log-variance

    kldiv_loss = kl_sum / x_dim
    loss = data_loss + kldiv_loss
    # ========================

    return loss, data_loss, kldiv_loss

