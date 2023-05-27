import torch
from torch import nn

class Generator(nn.Module):
    """
        This class implements a generator model that takes a latent space vector (z) as input 
        and produces a fake image that resembles the real images from the training dataset.

        The generator consists of several layers of transposed convolutions (also known as deconvolutions), 
        each followed by a batch normalization and a ReLU activation function. The last layer uses a Tanh 
        activation function to output pixel values in the range [-1, 1].

        Attributes:
            gen (torch.nn.Module): The sequential container of all the layers in the generator.

        Args:
            z_dim (int): The dimension of the latent space vector (z).
            channels_img (int): The number of output channels, typically 3 for RGB images.
            features_g (int): The size (number of nodes) of the first hidden layer in the generator.
    """
    def __init__(self, z_dim, channels_img, features_g):
        super(Generator, self).__init__() 
        self.gen = nn.Sequential(
            self.generator_block(z_dim, features_g * 16, 4, 1, 0), 
            self.generator_block(features_g * 16, features_g * 8, 4, 2, 1),  
            self.generator_block(features_g * 8, features_g * 4, 4, 2, 1),  
            self.generator_block(features_g * 4, features_g * 2, 4, 2, 1),  
            nn.ConvTranspose2d(
                features_g * 2, channels_img, kernel_size=4, stride=2, padding=1
            ),
            nn.Tanh(),
        )

    def generator_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    
    def forward(self, z):
        return self.gen(z)




