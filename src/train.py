import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from preprocess import data_loader
from preprocess import BATCH_SIZE,IMAGE_SIZE,CHANNELS_IMG
from generator import Generator
from discriminator import Discriminator


Z_DIM = 100 #dimension of the noise vector
NUM_EPOCHS  = 200 
LEARNING_RATE = 2e-4 #Adam learning rate
FEATURES_G = 64
FEATURES_D = 64

#Binary Cross Entropy Loss function
adversarial_loss = nn.BCELoss()

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
    for batch_idx, (images,_) in enumerate(data_loader):
        images = images.to(device)
        #-------------------------------------------------------
        #   Train Generator: max log(D(G(z)))
        #-------------------------------------------------------

        #--------------------------------------------------------
        #   Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        #--------------------------------------------------------

