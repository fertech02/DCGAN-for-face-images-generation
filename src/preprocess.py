from torchvision import datasets, transforms
from torch.utils.data import DataLoader

BATCH_SIZE = 128 #number of samples processed at iteration
IMAGE_SIZE = 64 #size of each image 
CHANNELS_IMG = 3 #number of channels in a rgb image

transforms = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE), #resize the image by 64x64
        transforms.ToTensor(), #transform the image in a pytorch tensor
        transforms.Normalize(
            [0.5 for i in range(CHANNELS_IMG)], [0.5 for i in range(CHANNELS_IMG)]) #ensure that every pixel is in range (-1,1)
    ]
)

#Preprocess the dataset
dataset = datasets.ImageFolder(root="data/img_align_celeba",transform=transforms)

#Create a dataloader
data_loader = DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True)

