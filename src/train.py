import os
import torch
import numpy as np
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
FEATURES_G = 64
FEATURES_D = 64

#Binary Cross Entropy Loss function
loss = nn.BCELoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Initialize generator and discriminator
generator = Generator(Z_DIM,CHANNELS_IMG,FEATURES_G).to(device)
discriminator = Discriminator(CHANNELS_IMG,FEATURES_D).to(device)

#Initialize optimizers for Generator and Discriminator
optimizer_g = optim.Adam(generator.parameters(),lr=LEARNING_RATE,betas=(0.5,0.999))
optimizer_d = optim.Adam(discriminator.parameters(),lr=LEARNING_RATE,betas=(0.5,0.999))

"""

    ---     TRAINING    ---

"""

for epoch in range(NUM_EPOCHS):
    for idx, (images,_) in enumerate(data_loader):

        images = images.to(device)
        real_labels = torch.ones_like(Tensor(BATCH_SIZE, 1)) #labels for real imgs
        fake_labels = torch.zeros_like(Tensor(BATCH_SIZE, 1)) #labels for fake imgs
        real_imgs = Variable(images.type(Tensor))

        #-------------------------------------------------------
        #   Train Generator: max log(D(G(z)))
        #-------------------------------------------------------

        optimizer_g.zero_grad() 
        z = Variable(Tensor(np.random.normal(0, 1, (BATCH_SIZE, Z_DIM, 1, 1)))) #noise vector
        g_images = generator(z) #generate a batch of images
        g_loss = loss(discriminator(g_images), real_labels) #generator loss
        g_loss.backward()
        optimizer_g.step()

        #--------------------------------------------------------
        #   Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        #--------------------------------------------------------

        optimizer_d.zero_grad() 
        real_loss = loss(discriminator(real_imgs), real_labels)
        fake_loss = loss(discriminator(g_images.detach()), fake_labels) 
        d_loss = real_loss + fake_loss #discriminator loss
        d_loss.backward()
        optimizer_d.step()

        if epoch == 10 or epoch % 50 == 0:
            os.mkdir(f"../results/{epoch} epochs")
            save_image(g_images.detach()[:20],f"../results/{epoch} epochs",nrow=5, normalize=True)
            


