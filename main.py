from image_proccessing import ImageProcessing
from criteria import MSE,PSNR,SNR,SSIM
img = ImageProcessing("./img/madrid-fullhd.jpg")
img.grayscale()
img.snp_noise()
img.wiener_filter()
# img.median_filter()
img.save("./img/snp-image.jpg")
img.show()
exit()
noised_img_value = noised_img.get_matrix_img
natural_img_value = natural_img.get_matrix_img
print("Mean Squared Error : {}".format(MSE(natural_img_value, noised_img_value)))
print("PSNR : {}".format(PSNR(natural_img_value, noised_img_value)))
print("SNR : {}".format(SNR(natural_img_value, noised_img_value)))
print("SSIM : {}".format(SSIM(natural_img_value, noised_img_value)))