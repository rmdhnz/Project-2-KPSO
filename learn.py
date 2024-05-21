import cv2 as cv
import numpy as np
from mynoise import NoiseForImage
img= NoiseForImage("./img/brain.jpg")
img_natural = NoiseForImage("./img/brain.jpg")
img.resize_image(500,500)
img_natural.resize_image(500,500)
img_natural.grayscale_image()
img.grayscale_image()
img.gaussian_noise(0,0.05)
# img.speckle_noise()
img_natural.display()
img.display("Noise image")
value_img = img.get_image_matrix()
value_natural_img = img_natural.get_image_matrix()
mse_value = np.mean(np.square(value_img-value_natural_img))
print(mse_value)
cv.waitKey(0)