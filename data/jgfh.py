
# import os
# import cv2
# print(cv2.__version__)

# #normalize the image
# def normalize_images(images):
#     normalized_images = []
#     for i in os.listdir(images):
#         image = cv2.imread(images+"/"+i)
#         #Convert the image in grayscale
#         if len(image.shape) == 3:
#           image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
#         #Normalize the pixel values in the range (0-1)
#         normalized_image = cv2.normalize(image,None, 0, 1,cv2.NORM_MINMAX)
#         normalized_images.append(normalized_image)
#     return normalize_images

# #Normalized Dataset
# normalized_dataset = normalize_images("data/img_align_celeba/img_align_celeba")
# cv2.imshow("Jerry",normalized_dataset[0])
# cv2.waitKey(0)
print("hi")
