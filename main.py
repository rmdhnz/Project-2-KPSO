from image_proccessing import ImageProcessing
img = ImageProcessing("./img/madrid-fullhd.jpg")
img.grayscale()
img.gaussian_noise(0,0.05)
img.wiener_filter()
img.wiener_filter()
img.wiener_filter()
img.wiener_filter()
  # img.save("./img/mmwf-snp.jpg")
img.show()
exit()
try :
  pass
except RuntimeWarning : 
  print("Can't divide by zero")
except : 
  print("Something went wrong")
finally : 
  print("Done  :)")