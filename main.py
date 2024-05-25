from image_proccessing import ImageProcessing
from criteria import MSE,PSNR,SNR,SSIM

img = ImageProcessing("./img/madrid-fullhd.jpg")
img.grayscale()
img.snp_noise()
img.median_filter()
img.show()
img.save("./img/median-snp.jpg")