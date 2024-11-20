import torch.nn as nn

class WGAN_Generator(nn.Module):
    def __init__(self, ngpu, latent_dim, feature_maps_gen, num_channels, inplace=False):
        super(WGAN_Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(latent_dim, feature_maps_gen * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_maps_gen * 8),
            nn.ReLU(inplace=inplace),
            # state size. ``(feature_maps_gen*8) x 4 x 4``
            nn.ConvTranspose2d(feature_maps_gen * 8, feature_maps_gen * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps_gen * 4),
            nn.ReLU(inplace=inplace),
            # state size. ``(feature_maps_gen*4) x 8 x 8``
            nn.ConvTranspose2d(feature_maps_gen * 4, feature_maps_gen * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps_gen * 2),
            nn.ReLU(inplace=inplace),
            # state size. ``(feature_maps_gen*2) x 16 x 16``
            nn.ConvTranspose2d(feature_maps_gen * 2, feature_maps_gen, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps_gen),
            nn.ReLU(inplace=inplace),
            # state size. ``(feature_maps_gen) x 32 x 32``
            nn.ConvTranspose2d(feature_maps_gen, num_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. ``(num_channels) x 64 x 64``
        )

    def forward(self, input):
        return self.main(input)
    

class WGAN_Critic(nn.Module):
    def __init__(self, ngpu, feature_maps_critic, num_channels, inplace=False):
        super(WGAN_Critic, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is ``(num_channels) x 64 x 64``
            nn.Conv2d(num_channels, feature_maps_critic, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=inplace),
            # state size. ``(feature_maps_critic) x 32 x 32``
            nn.Conv2d(feature_maps_critic, feature_maps_critic * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps_critic * 2),
            nn.LeakyReLU(0.2, inplace=inplace),
            # state size. ``(feature_maps_critic*2) x 16 x 16``
            nn.Conv2d(feature_maps_critic * 2, feature_maps_critic * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps_critic * 4),
            nn.LeakyReLU(0.2, inplace=inplace),
            # state size. ``(feature_maps_critic*4) x 8 x 8``
            nn.Conv2d(feature_maps_critic * 4, feature_maps_critic * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps_critic * 8),
            nn.LeakyReLU(0.2, inplace=inplace),
            # state size. ``(feature_maps_critic*8) x 4 x 4``
            nn.Conv2d(feature_maps_critic * 8, 1, 4, 1, 0, bias=False)
        )

    def forward(self, input):
        return self.main(input)
    

# Generator Code
class dcGAN_Generator(nn.Module):
    def __init__(self, ngpu, latent_dim, feature_maps_gen, num_channels, inplace=False):
        super(dcGAN_Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( latent_dim, feature_maps_gen * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_maps_gen * 8),
            nn.ReLU(inplace=inplace),
            # state size. ``(feature_maps_gen*8) x 4 x 4``
            nn.ConvTranspose2d(feature_maps_gen * 8, feature_maps_gen * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps_gen * 4),
            nn.ReLU(inplace=inplace),
            # state size. ``(feature_maps_gen*4) x 8 x 8``
            nn.ConvTranspose2d( feature_maps_gen * 4, feature_maps_gen * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps_gen * 2),
            nn.ReLU(inplace=inplace),
            # state size. ``(feature_maps_gen*2) x 16 x 16``
            nn.ConvTranspose2d( feature_maps_gen * 2, feature_maps_gen, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps_gen),
            nn.ReLU(inplace=inplace),
            # state size. ``(feature_maps_gen) x 32 x 32``
            nn.ConvTranspose2d( feature_maps_gen, num_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. ``(nc) x 64 x 64``
        )

    def forward(self, input):
        return self.main(input)
    

class dcGAN_Discriminator(nn.Module):
    def __init__(self, ngpu, feature_maps_critic, num_channels, inplace=False):
        super(dcGAN_Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is ``(num_channels) x 64 x 64``
            nn.Conv2d(num_channels, feature_maps_critic, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=inplace),
            # state size. ``(feature_maps_critic) x 32 x 32``
            nn.Conv2d(feature_maps_critic, feature_maps_critic * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps_critic * 2),
            nn.LeakyReLU(0.2, inplace=inplace),
            # state size. ``(feature_maps_critic*2) x 16 x 16``
            nn.Conv2d(feature_maps_critic * 2, feature_maps_critic * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps_critic * 4),
            nn.LeakyReLU(0.2, inplace=inplace),
            # state size. ``(feature_maps_critic*4) x 8 x 8``
            nn.Conv2d(feature_maps_critic * 4, feature_maps_critic * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps_critic * 8),
            nn.LeakyReLU(0.2, inplace=inplace),
            # state size. ``(feature_maps_critic*8) x 4 x 4``
            nn.Conv2d(feature_maps_critic * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)