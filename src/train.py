import os
import torch
import numpy as nn
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.autograd import Variable
from torchvision.utils import save_image
from preprocess import data_loader,BATCH_SIZE,CHANNELS_IMG
from generator import Generator
from discriminator import Discriminator


Z_DIM = 100 #dimension of the noise vector
NUM_EPOCHS  = 200 
LEARNING_RATE = 2e-4 #Adam learning rate
FEATURES_G = 64 #complexity of generator
FEATURES_D = 64 #complexity of discriminator

#Binary Cross Entropy Loss function
loss_function = nn.BCELoss()

#function to initialize weights
def initialize_weights(model):
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
            nn.init.normal_(module.weight.data, mean=0.0, std=0.02)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.normal_(module.weight.data, mean=1.0, std=0.02)
            nn.init.constant_(module.bias.data, 0.0)
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Initialize generator and discriminator
generator = Generator(Z_DIM,CHANNELS_IMG,FEATURES_G).to(device)
discriminator = Discriminator(CHANNELS_IMG,FEATURES_D).to(device)

#Initialize weights
initialize_weights(generator)
initialize_weights(discriminator)

#Initialize optimizers for Generator and Discriminator
optimizer_g = optim.Adam(generator.parameters(),lr=LEARNING_RATE,betas=(0.5,0.999))
optimizer_d = optim.Adam(discriminator.parameters(),lr=LEARNING_RATE,betas=(0.5,0.999))

"""

    ---     TRAINING    ---

"""

for epoch in range(NUM_EPOCHS+1):
    for batch_idx, (images,_) in enumerate(data_loader):

        images = images.to(device)
        real_labels = torch.ones_like(Tensor(BATCH_SIZE, 1)).to(device) #labels for real imgs
        fake_labels = torch.zeros_like(Tensor(BATCH_SIZE, 1)).to(device) #labels for fake imgs
        real_imgs = Variable(images.type(Tensor)).to(device)

        #-------------------------------------------------------
        #   Train Generator: max log(D(G(z)))
        #-------------------------------------------------------

        optimizer_g.zero_grad() 
        noise_vector = torch.randn(BATCH_SIZE,Z_DIM, 1, 1).to(device)
        #noise_vector = noise_vector.unsqueeze(-1).unsqueeze(-1)
        g_images = generator(noise_vector) #generate a batch of images
        #g_loss = loss_function(discriminator(g_images.detach()).squeeze(-1).squeeze(-1), real_labels).to(device) #generator loss
        g_loss = loss_function(discriminator(g_images.detach(), real_labels)).to(device)
        g_loss.backward(retain_graph=True)
        optimizer_g.step()

        #--------------------------------------------------------
        #   Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        #--------------------------------------------------------

        optimizer_d.zero_grad() 
        #real_loss = loss_function(discriminator(real_imgs).squeeze().unsqueeze(-1), real_labels)
        real_loss = loss_function(discriminator(real_imgs), real_labels)
        #fake_loss = loss_function(discriminator(g_images.detach()).squeeze().unsqueeze(-1), fake_labels) 
        fake_loss = loss_function(discriminator(g_images.detach()), fake_labels)
        d_loss = real_loss + fake_loss #discriminator loss
        d_loss.backward()
        optimizer_d.step()
        with torch.no_grad():
            if epoch != 0 and (epoch == 10 or epoch % 50 == 0):
                path_to_save = f"/content/DCGAN-for-face-images-generation/results/{epoch} epochs"
                try:
                    os.makedirs(path_to_save)
                except FileExistsError:
                    pass
                for i, image_tensor in enumerate(g_images[-25::]):
                    save_image(image_tensor, os.path.join(path_to_save, f'image_{i}.png'))
                save_image(g_images[:25], os.path.join(path_to_save, 'grid.png'), nrow=8, normalize=True)
            


