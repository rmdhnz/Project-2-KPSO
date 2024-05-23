from scipy.signal import convolve2d
from skimage import restoration,color
from skimage.io import imread
import matplotlib.pyplot as plt
from skimage.util import random_noise
import numpy as np
from image_proccessing import ImageProcessing

img = ImageProcessing("./img/squad-madrid.jpeg")
img.grayscale()
img.gaussian_noise(mean=0,var=0.05)
img.gaussian_blur()
img.save("./gaussian.jpg")
img.show()
value_img = img.get_matrix_img
print(value_img.shape)
exit()

def display(image,title=None) : 
  plt.imshow(image,cmap="gray")
  plt.axis('off')
  plt.title(title)
  plt.show()

image = imread("./img/brain.jpg")
img_gray = color.rgb2gray(image)
k = 5
psf = np.ones((k,k))/(k**2)
img_noise = convolve2d(img_gray,psf,"same")
img_noise = random_noise(img_noise,mode="gaussian",mean=0,var=0.1)

print(img_noise)
imgRestor= restoration.wiener(img_noise,psf=psf,balance=0.35)
plt.subplot(1,2,1)
plt.imshow(img_noise,cmap="gray")
plt.title("Gambar noise")
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(imgRestor,cmap="gray")
plt.axis('off')
plt.title("Gambar filter")
plt.show()
