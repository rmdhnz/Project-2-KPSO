import numpy as np
import cv2 as cv

blank = np.ones((500,500),dtype="uint8")
blank*=255
print(blank)
cv.imshow("blank", blank)
cv.waitKey(0)