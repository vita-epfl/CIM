"""model.py"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

from oatomobile.baselines.torch.cim.perception.types_ import *
from oatomobile.baselines.torch import transforms
from oatomobile.core.typing import ShapeLike

from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

from oatomobile.baselines.torch import transforms
from oatomobile.baselines.torch.models import MLP
from oatomobile.baselines.torch.models import MobileNetV2
from oatomobile.baselines.torch.typing import ArrayLike
from oatomobile.core.typing import ShapeLike



def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std*eps


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class BetaVAE(nn.Module):
    """Model proposed in original beta-VAE paper(Higgins et al, ICLR, 2017)."""

    def __init__(self, latent_dim=10, channels=3, beta=10, dist='bernoulli'):
        super(BetaVAE, self).__init__()
        self.latent_dim = latent_dim
        self.channels = channels
        self.beta = beta
        self.dist = dist

        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 32, 4, 2, 1),
            nn.LeakyReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.LeakyReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.LeakyReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.LeakyReLU(True),
            View((-1, 64 * 4 * 4)),
            nn.Linear(64 * 4 * 4, 256),
            nn.LeakyReLU(True),
            nn.Linear(256, 256),
            nn.LeakyReLU(True),
            nn.Linear(256, latent_dim * 2),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(True),
            nn.Linear(256, 256),
            nn.LeakyReLU(True),
            nn.Linear(256, 64 * 4 * 4),
            nn.LeakyReLU(True),
            View((-1, 64, 4, 4)),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(32, channels, 4, 2, 1),
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x):
        distributions = self._encode(x)
        mu = distributions[:, :self.latent_dim]
        logvar = distributions[:, self.latent_dim:]
        z = reparametrize(mu, logvar)
        x_recon = self._decode(z)

        return x_recon, x, mu, logvar

    def _encode(self, x):
        return self.encoder(x)

    def encode(self, x):
        distributions = self._encode(x)
        mu = distributions[:, :self.latent_dim]
        return mu

    def _decode(self, z):
        return self.decoder(z)

    def decode(self, z):
        return F.sigmoid(self._decode(z))

    def loss(self, x_recon, x, mu, log_var):
        recon_loss = reconstruction_loss(x, x_recon, self.dist)
        total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, log_var)
        loss = recon_loss + self.beta*total_kld
        return {'loss': loss, 'Reconstruction_Loss': recon_loss, 'KLD': total_kld}

    def transform(
            self,
            sample: Mapping[str, ArrayLike],
    ) -> Mapping[str, torch.Tensor]:
        """Prepares variables for the interface of the model.
        Args:
          sample: (keyword arguments) The raw sample variables.
        Returns:
          The processed sample.
        """

        # Renames `bird_view_camera_cityscapes` to `visual_features`.
        if "bird_view_camera_cityscapes" in sample:
            sample["visual_features"] = sample.pop("bird_view_camera_cityscapes")

        # Preprocesses the visual features.
        if "visual_features" in sample:
            img = sample["visual_features"]
            device = next(self.parameters()).device
            img = transforms.convert_color(img, (0.41960785, 0.5568628, 0.13725491), (0, 0, 0), device)
            img = transforms.convert_color(img, (0, 0, 0.5568628), (1, 1, 1), device)
            img = transforms.convert_color(img, (0.27450982, 0.27450982, 0.27450982), (0, 0, 0), device)
            img = transforms.convert_color(img, (0.98039216, 0.6666667, 0.627451), (0, 0, 0), device)
            img = transforms.convert_color(img, (0.4, 0.4, 0.6117647), (0, 0, 0), device)
            img = transforms.convert_color(img, (0.74509805, 0.6, 0.6), (0, 0, 0), device)
            img = transforms.convert_color(img, (0.8627451, 0.07843138, 0.23529412), (0, 0, 0), device)
            img = transforms.convert_color(img, (0.5019608, 0.2509804, 0.5019608), (0, 0, 0), device)
            img = transforms.convert_color(img, (0.6, 0.6, 0.6), (0, 0, 0), device)
            img = transforms.convert_color(img, (0.95686275, 0.13725491, 0.9098039), (0, 0, 0), device)
            img = transforms.convert_color(img, (0.6156863, 0.91764706, 0.19607843), (0, 0, 0), device)
            img = transforms.convert_color(img, (0.8627451, 0.8627451, 0), (0, 0, 0), device)
            img = transforms.RGB_to_gray(img, device)
            sample["visual_features"] = transforms.transpose_visual_features(
                transforms.downsample_visual_features(
                    visual_features=img,
                    output_shape=(64, 64),
                ))

        return sample

    def reconstruct(self, x, **kwargs):
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        z = self.encode(x)
        return self.decode(z)

def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5*(1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld

def reconstruction_loss(x, x_recon, distribution):
    batch_size = x.size(0)
    assert batch_size != 0

    if distribution == 'bernoulli':
        recon_loss = F.binary_cross_entropy_with_logits(x_recon, x, size_average=False).div(batch_size)
    elif distribution == 'gaussian':
        x_recon = F.sigmoid(x_recon)
        recon_loss = F.mse_loss(x_recon, x, size_average=False).div(batch_size)
    else:
        recon_loss = None

    return recon_loss

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()