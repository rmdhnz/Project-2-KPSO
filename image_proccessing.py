from skimage import io
import numpy as np
import cv2 as cv
from skimage.filters import median,butterworth
from skimage import restoration,color
import matplotlib.pyplot as plt
from skimage.util import random_noise

class ImageProcessing : 
  def __init__(self,file_location) : 
    self.img = io.imread(file_location)
    self.__k=3
    self.__psf = np.ones((self.__k,self.__k))/(self.__k**2)
  
  def shows(*objs) : 
    k=1
    for obj in objs : 
      plt.subplot(1,len(objs),k)
      k+=1
      plt.imshow(obj.img,cmap="gray")
      plt.axis('off')
    plt.show()
  
  def show(self) :
    plt.imshow(self.img,cmap="gray")
    plt.axis('off')
    plt.show()
  
  def grayscale(self) : 
    self.img = color.rgb2gray(self.img)

  def speckle_noise(self,mean=None,var=None) : 
    if mean is None or var is None :
      self.img = random_noise(self.img,mode="speckle")
    else:
      self.img = random_noise(self.img,mode="speckle",mean=mean,var=var)
  def poisson_noise(self) : 
    self.img = random_noise(self.img,mode="speckle",mean=0,var=0.05)
  
  def snp_noise(self) :
    self.img = random_noise(self.img,mode="s&p")
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
    self.img = restoration.wiener(self.img,psf=self.__psf,balance=0.5)
  def median_filter(self) : 
    self.img= median(self.img)
  def butterworth_filter(self,orde=2) : 
    self.img = butterworth(self.img,order=orde)
  @property
  def get_matrix_img(self) : 
    return self.img
