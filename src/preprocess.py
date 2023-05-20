from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import os

#Normalize the images
def normalize_images(images):
    normalized_images = []
    for i in os.listdir(images):
        image = cv2.imread(images+'/'+i)
        #Convert all the images in a range (0-1)
        image = image.astype(np.float32)
        normalized_image = image / 255.0
        normalized_images.append(normalized_image)
    return normalized_images

normalized_dataset = normalize_images("data/img_align_celeba/img_align_celeba")

#Shuffle and Split the dataset
train_data, val_test_data = train_test_split(normalized_dataset,test_size=0.2,random_state=42)
val_data, test_data = train_test_split(val_test_data, test_size=0.5,random_state=42)

#Batches Creation
batch_size = 32

def create_batches(data, batch_s):
    indices = np.arange(len(data))
    for start_idx in range(0,len(data),batch_s):
        end_idx = min(start_idx + batch_s, len(data))
        batch_indices = indices[start_idx:end_idx]
        batch = data[batch_indices]
        yield batch

train_batches = create_batches(train_data,batch_size)
val_batches = create_batches(val_data,batch_size)
