import torch
from torch import nn

class Generator(nn.Module):
    """
        
    """
    def __init__(self, z_dim, channels_img, features_g):
        super(Generator, self).__init__() 
        self.gen = nn.Sequential(
            # Input: N x 100 x 1 x 1
            self.generator_block(z_dim, features_g * 16, 4, 1, 0), 
            self.generator_block(features_g * 16, features_g * 8, 4, 2, 1),  
            self.generator_block(features_g * 8, features_g * 4, 4, 2, 1),  
            self.generator_block(features_g * 4, features_g * 2, 4, 2, 1),  
            nn.ConvTranspose2d(
                features_g * 2, channels_img, kernel_size=4, stride=2, padding=1
            ),  
            # Output: N x 3 x 64 x 64
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
        return self.z


