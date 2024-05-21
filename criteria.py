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
  return structural_similarity(original,predict)
