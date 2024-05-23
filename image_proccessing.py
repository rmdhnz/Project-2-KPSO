from skimage import io
import numpy as np
import cv2 as cv
from skimage import restoration,color
import matplotlib.pyplot as plt
from skimage.util import random_noise

class ImageProcessing : 
  def __init__(self,file_location) : 
    self.img = io.imread(file_location)
    self.__k=5
    self.__psf = np.ones((self.__k,self.__k))/(self.__k**2)
  
  def show(self) :
    plt.imshow(self.img,cmap="gray")
    plt.axis('off')
    plt.show()
  
  def grayscale(self) : 
    self.img = color.rgb2gray(self.img)
  
  def gaussian_noise(self,mean=None,var=None) :
    if mean is None or var is None :
      self.img = random_noise(self.img,mode="gaussian")
    else:
      self.img = random_noise(self.img,mode="gaussian",mean=mean,var=var)
  
  def gaussian_blur(self) : 
    self.img = cv.GaussianBlur(self.img,(7,7),0)
  def save(self,filename) : 
    gambar = self.img
    gambar*=255
    gambar = gambar.astype("uint8")
    io.imsave(filename,gambar)
  def wiener_filter(self) :
    self.img = restoration.wiener(self.img,psf=self.__psf,balance=0.35)
  @property
  def get_matrix_img(self) : 
    return self.img
