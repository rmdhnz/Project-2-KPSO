from image_proccessing import ImageProcessing
from criteria import MSE,PSNR,SNR,SSIM
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
img = ImageProcessing("./img/madrid-fullhd.jpg")

img.grayscale()
img.gaussian_noise(0,0.05)
img.butterworth_filter()
img.img*=255
img.img = np.array(img.img)
r,c=0,0
for i in img.img : 
  for j in i : 
    j=0
    if j <0 : 
      img.img[r][c] = 0
    c+=1  
  r+=1

print(img.img)
img.show()
print(img.img.shape)