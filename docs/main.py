import os
import cv2



# Normalize the images
def normalize_images(images):
    normalized_images = []
    for i in os.listdir(images):
        image = cv2.imread(os.path.join(images, i))
        if image is not None:
            # Convert the image to grayscale
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Normalize the pixel values in the range (0-1)
            normalized_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
            normalized_images.append(normalized_image)
    return normalized_images

# Normalized Dataset
normalized_dataset = normalize_images("data/img_align_celeba/img_align_celeba")

if len(normalized_dataset) > 0:
    # Display the first normalized image
    cv2.imshow("Jerry", normalized_dataset[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

