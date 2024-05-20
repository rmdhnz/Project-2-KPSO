import cv2 as cv
import numpy as np
from mynoise import NoiseForImage

blank = np.ones((500,500),dtype="uint8")
my_img = NoiseForImage("./img/Loose1.png")
my_img.resize_image(500,500)
my_img.grayscale_image()
my_img.gaussian_noise(0,0.1)
# my_img.speckle_noise()
# my_img.poisson_noise()
my_img.display("Image")
print(my_img.image)

cv.waitKey(0)