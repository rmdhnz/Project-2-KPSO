import cv2 as cv

img = cv.imread("./img/Loose1.png")
img_gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow("Image",img)

def changeRes(width,height) :
  capture.set(3,width)
  capture.set(4,height)
  

def rescaleFrame(frame,scale=0.75) : 
  width = int(frame.shape[0]*scale)
  height = int(frame.shape[1]*scale)
  dimensions = (width, height)

  return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)

capture = cv.VideoCapture(0)
while True : 
  isTrue,frame = capture.read()
  frame_resized = rescaleFrame(frame)
  cv.imshow("Capture",frame)
  cv.imshow("Resized Capture",frame_resized)

  if  cv.waitKey(20) & 0xff==ord('q') : 
    break

capture.release()
cv.destroyAllWindows()

cv.waitKey(0)