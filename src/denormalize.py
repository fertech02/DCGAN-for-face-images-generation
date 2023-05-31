import cv2
import numpy as np

image = cv2.imread('results/150 epochs/image_20.png')
image = ((image + 1) * 255).astype('uint8')
image = cv2.resize(image, (400,400))
cv2.imshow("Image",image)
cv2.waitKey(0)