import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

BATCH_SIZE = 128 #number of samples processed at iteration
IMAGE_SIZE = 64 #size of each image 
CHANNELS_IMG = 3 #number of channels in a rgb image

class CelebaDataset(Dataset):

    def __init__(self,root_dir,transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_list = os.listdir(root_dir)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_list[idx])
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image



transforms = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE), #resize the image by 64x64
        transforms.ToTensor(), #transform the image in a pytorch tensor
        transforms.Normalize(
            [0.5 for i in range(CHANNELS_IMG)], [0.5 for i in range(CHANNELS_IMG)]) #ensure that every pixel is in range (-1,1)
    ]
)
dataset = CelebaDataset(root_dir="../img_align_celeba",transform=transforms)
#Preprocess the dataset
#dataset = datasets.ImageFolder(root="../img_align_celeba",transform=transforms)

#Create a dataloader
data_loader = DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True)

