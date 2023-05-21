from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split

BATCH_SIZE = 128 #number of samples processed at time
IMAGE_SIZE = 64 #size of each image 
CHANNELS_IMG = 3 #number of channels in a rgb image


transforms = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for i in range(CHANNELS_IMG)], [0.5 for i in range(CHANNELS_IMG)])
    ]
)

#Load and preprocess the dataset
dataset = datasets.ImageFolder(root="data/img_align_celeba",transform=transforms)

#Split the dataset
train_size = int(0.7*len(dataset)) #size of the training set
val_size = int(0.15*len(dataset)) #size of the validation set
test_size = len(dataset) - train_size - val_size #size of the test set

#Create the subsets
train_data, val_data, test_data = random_split(dataset, [train_size, val_size, test_size])

#Create a dataloader for each subsets
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
