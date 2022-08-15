import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Callable
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer

from hw4.scores import samples_to_fid
class Discriminator(nn.Module):
    def __init__(self, in_size):
        """
        :param in_size: The size of on input image (without batch dimension).
        """
        super().__init__()
        self.in_size = in_size
        # TODO: Create the discriminator model layers.
        #  To extract image features you can use the EncoderCNN from the VAE
        #  section or implement something new.
        #  You can then use either an affine layer or another conv layer to
        #  flatten the features.
        # ====== YOUR CODE: ======
        layers = []
        in_channels = self.in_size[0]
        out_channels = 1024
        channels = [64, 256, 512,out_channels]
        kernel_size = 4
        stride = 2
        padding = 1
        for out_channel in channels:
            conv_layer = nn.Conv2d(in_channels, out_channel, kernel_size, stride, padding,bias=False)
            norm_layer = nn.BatchNorm2d(out_channel)
            activation_layer = nn.LeakyReLU(0.2, inplace=True)

            layers.append(conv_layer)
            layers.append(norm_layer)
            layers.append(activation_layer)

            in_channels = out_channel
        layers.append(nn.Conv2d(out_channel,1,kernel_size,1,0,bias=False))

        self.discriminator = nn.Sequential(*layers)
        # ========================

    def forward(self, x):
        """
        :param x: Input of shape (N,C,H,W) matching the given in_size.
        :return: Discriminator class score (not probability) of
        shape (N,).
        """
        # TODO: Implement discriminator forward pass.
        #  No need to apply sigmoid to obtain probability - we'll combine it
        #  with the loss due to improved numerical stability.
        # ====== YOUR CODE: ======
        y = torch.flatten(self.discriminator(x),1)
        # ========================
        return y

class SpectralNormalizationDiscriminator(nn.Module):

    def __init__(self, in_size):
        super().__init__()

        self.in_size = in_size

        layers = []
        in_channels = self.in_size[0]
        out_channels = 1024
        channels = [64, 256, 512, out_channels]
        kernel_size = 4
        stride = 2
        padding = 1
        for out_channel in channels:
            conv_layer = torch.nn.utils.parametrizations.\
                spectral_norm(nn.Conv2d(in_channels, out_channel, kernel_size, stride, padding, bias=False))
            activation_layer = nn.LeakyReLU(0.2, inplace=True)

            layers.append(conv_layer)
            layers.append(activation_layer)

            in_channels = out_channel
        layers.append(nn.Conv2d(out_channel, 1, kernel_size, 1, 0, bias=False))

        self.discriminator = nn.Sequential(*layers)

    def forward(self,x):
        y = torch.flatten(self.discriminator(x),1)
        return y

class Generator(nn.Module):
    def __init__(self, z_dim, featuremap_size=4, out_channels=3):
        """
        :param z_dim: Dimension of latent space.
        :featuremap_size: Spatial size of first feature map to create
        (determines output size). For example set to 4 for a 4x4 feature map.
        :out_channels: Number of channels in the generated image.
        """
        super().__init__()
        self.z_dim = z_dim

        # TODO: Create the generator model layers.
        #  To combine image features you can use the DecoderCNN from the VAE
        #  section or implement something new.
        #  You can assume a fixed image size.
        # ====== YOUR CODE: ======
        #hint (you dont have to use....)
        #from .autoencoder import DecoderCNN
        ngf = 64
        self.cnn = nn.Sequential(

            nn.ConvTranspose2d(self.z_dim, ngf * 8, featuremap_size, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d( ngf, out_channels, 4, 2, 1, bias=False),
            nn.Tanh()

        )
        # ========================

    def sample(self, n, with_grad=False):
        """
        Samples from the Generator.
        :param n: Number of instance-space samples to generate.
        :param with_grad: Whether the returned samples should be part of the
        generator's computation graph or standalone tensors (i.e. should be
        be able to backprop into them and compute their gradients).
        :return: A batch of samples, shape (N,C,H,W).
        """
        device = next(self.parameters()).device
        # TODO: Sample from the model.
        #  Generate n latent space samples and return their reconstructions.
        #  Don't use a loop.
        # ====== YOUR CODE: ======
        z = torch.randn((n, self.z_dim, 1, 1), device= device)

        if not with_grad:
            with torch.no_grad():
                return self.cnn(z)
        else:
            samples = self.cnn(z)
        # ========================
        return samples.to(device)

    def forward(self, z):
        """
        :param z: A batch of latent space samples of shape (N, latent_dim).
        :return: A batch of generated images of shape (N,C,H,W) which should be
        the shape which the Discriminator accepts.
        """
        # TODO: Implement the Generator forward pass.
        #  Don't forget to make sure the output instances have the same
        #  dynamic range as the original (real) images.
        # ====== YOUR CODE: ======
        z = z.view(-1, self.z_dim, 1, 1)
        x = self.cnn(z)
        # ========================
        return x


def discriminator_loss_fn(y_data, y_generated, data_label=0, label_noise=0.0):
    """
    Computes the combined loss of the discriminator given real and generated
    data using a binary cross-entropy metric.
    This is the loss used to update the Discriminator parameters.
    :param y_data: Discriminator class-scores of instances of data sampled
    from the dataset, shape (N,).
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :param label_noise: The range of the noise to add. For example, if
    data_label=0 and label_noise=0.2 then the labels of the real data will be
    uniformly sampled from the range [-0.1,+0.1].
    :return: The combined loss of both.
    """
    assert data_label == 1 or data_label == 0
    # TODO:
    #  Implement the discriminator loss. Apply noise to both the real data and the
    #  generated labels.
    #  See pytorch's BCEWithLogitsLoss for a numerically stable implementation.
    # ====== YOUR CODE: ======

    N = y_data.shape[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    noize_data = random_noise(N,label_noise).to(device)

    noize_generated = random_noise(N, label_noise).to(device)

    generated_label = 1 if data_label == 0 else 0  # set generated_label to opposite of data_label

    data_classification = torch.full((N,), data_label, dtype=torch.float, device = device) + noize_data

    generated_classification = torch.full((N,), generated_label, dtype=torch.float, device = device) + noize_generated

    loss_data = torch.nn.functional.binary_cross_entropy_with_logits(y_data, data_classification).to(device)

    loss_generated = torch.nn.functional.binary_cross_entropy_with_logits(y_generated, generated_classification)

    # ========================
    return loss_data + loss_generated


def generator_loss_fn(y_generated, data_label=0):
    """
    Computes the loss of the generator given generated data using a
    binary cross-entropy metric.
    This is the loss used to update the Generator parameters.
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :return: The generator loss.
    """
    assert data_label == 1 or data_label == 0
    # TODO:
    #  Implement the Generator loss.
    #  Think about what you need to compare the input to, in order to
    #  formulate the loss in terms of Binary Cross Entropy.
    # ====== YOUR CODE: ======
    N = y_generated.shape[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_samples = torch.full((N,), data_label, dtype=torch.float, device=device)

    loss = torch.nn.functional.binary_cross_entropy_with_logits(y_generated, data_samples)
    # ========================
    return loss


def train_batch(
    dsc_model: Discriminator,
    gen_model: Generator,
    dsc_loss_fn: Callable,
    gen_loss_fn: Callable,
    dsc_optimizer: Optimizer,
    gen_optimizer: Optimizer,
    x_data: Tensor,
):
    """
    Trains a GAN for over one batch, updating both the discriminator and
    generator.
    :return: The discriminator and generator losses.
    """

    # TODO: Discriminator update
    #  1. Show the discriminator real and generated data
    #  2. Calculate discriminator loss
    #  3. Update discriminator parameters
    # ====== YOUR CODE: ======
    dsc_model.zero_grad()

    n = x_data.shape[0]

    dsc_y_data = torch.flatten(dsc_model(x_data))

    fake_samples = gen_model.sample(n)

    dsc_y_generated = torch.flatten(dsc_model(fake_samples))

    dsc_loss = dsc_loss_fn(dsc_y_data, dsc_y_generated)

    dsc_loss.backward()

    dsc_optimizer.step()

    # ========================

    # TODO: Generator update
    #  1. Show the discriminator generated data
    #  2. Calculate generator loss
    #  3. Update generator parameters
    # ====== YOUR CODE: ======
    gen_model.zero_grad()

    fake_samples = gen_model.sample(n,True)

    dsc_output = torch.flatten(dsc_model(fake_samples))

    gen_loss = gen_loss_fn(dsc_output)

    gen_loss.backward()

    gen_optimizer.step()
    # ========================

    return dsc_loss.item(), gen_loss.item()


def save_checkpoint(gen_model, dsc_losses, gen_losses, checkpoint_file):
    """
    Saves a checkpoint of the generator, if necessary.
    :param gen_model: The Generator model to save.
    :param dsc_losses: Avg. discriminator loss per epoch.
    :param gen_losses: Avg. generator loss per epoch.
    :param checkpoint_file: Path without extension to save generator to.
    """

    saved = False
    checkpoint_file = f"{checkpoint_file}.pt"

    # TODO:
    #  Save a checkpoint of the generator model. You can use torch.save().
    #  You should decide what logic to use for deciding when to save.
    #  If you save, set saved to True.
    # ====== YOUR CODE: ======
    epoch_number = len(dsc_losses)
    if epoch_number > 1:
        curr_gen_loss = gen_losses[-1]

        best_gen_loss = min(gen_losses[0:-1])

        if curr_gen_loss < best_gen_loss:
            torch.save(gen_model.state_dict, checkpoint_file)
            saved = True
            print(f"saved model at epoch number {epoch_number} !")
    # ========================

    return saved


def random_noise(n, label_noise):

    lower = -label_noise / 2

    dist = torch.distributions.uniform.Uniform(lower,-lower)

    noise = dist.sample((n,))

    return noise
