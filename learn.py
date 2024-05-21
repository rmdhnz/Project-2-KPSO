import cv2 as cv
from mynoise import NoiseForImage
img = NoiseForImage("./img/Loose1.png")
img.resize_image(500,500)
img.grayscale_image()
img.gaussian_noise(0,0.1)
img.wiener_filter()
img.display()
cv.waitKey(0)