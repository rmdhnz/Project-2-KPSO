import numpy as np
from skimage.metrics import mean_squared_error,peak_signal_noise_ratio,structural_similarity
def SNR(actual,pred) : 
  actual,pred = np.array(actual),np.array(pred)
  num = np.sum(np.square(actual))
  den = np.sum(np.square(pred-actual))
  return 10*(num/den)

def PSNR(original, predict): 
    original,predict = np.array(original),np.array(predict)
    return peak_signal_noise_ratio(original,predict)

def MSE(original, predict):
  original,predict = np.array(original),np.array(predict)
  return mean_squared_error(original,predict)

def SSIM(original,predict): 
  original,predict = np.array(original),np.array(predict)
  max_val = max(original.max(),predict.max())
  min_val = min(original.min(),predict.min())
  return structural_similarity(original,predict,data_range=max_val-min_val)

if __name__ == "__main__":
  from image_proccessing import ImageProcessing
  gray_img  = ImageProcessing("./gray_img.jpg")
  gauss_img = ImageProcessing("./img/mmwf-gauss.jpg")
  ImageProcessing.shows(gray_img,gauss_img)
  print("MSE : {:.2f}\n".format(MSE(gray_img.img,gauss_img.img)))
  print("PSNR : {:.2f}\n".format(PSNR(gray_img.img,gauss_img.img)))
  print("SNR : {:.2f}\n".format(SNR(gray_img.img,gauss_img.img)))
  print("SSIM : {:.2f}\n".format(SSIM(gray_img.img,gauss_img.img)))