import cv2 as cv
# import numpy as np
from skimage.util import random_noise
from skimage import io
class NoiseForImage : 

  def __init__(self,image_location) : 
    self.image = cv.imread(image_location)
  def resize_image(self,width,height) : 
    self.image=cv.resize(self.image,(width,height))
  def grayscale_image(self) : 
    self.image=cv.cvtColor(self.image,cv.COLOR_BGR2GRAY)
  
  def display(self,title) : 
    cv.imshow(title,self.image)
  
  def gaussian_noise(self,mean,var) :
    self.image = random_noise(self.image,mode="gaussian",mean=mean,var=var)
    self.image*=255
    self.image=self.image.astype('uint8')
  
  def speckle_noise(self) : 
    self.image = random_noise(self.image,mode="speckle")
  def poisson_noise(self) : 
    self.image = random_noise(self.image,mode="poisson")

  def save(self,filename) : 
    io.imsave(filename,self.image)