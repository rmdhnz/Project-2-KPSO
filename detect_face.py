import cv2 as cv
import os
import matplotlib.pyplot as plt
face_ref = cv.CascadeClassifier("face_ref.xml")
camera = cv.VideoCapture(0)

def face_detection(frame) : 
  optimized_frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
  faces = face_ref.detectMultiScale(optimized_frame,scaleFactor=1.1)
  return faces

def drawer_box(frame) :  
  for x,y,w,h in face_detection(frame) :
    cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),4)

def close_window() :
  camera.release()
  cv.destroyAllWindows()
  exit()
def main() :
  try : 
    if os.path.exists(image_location := "./snp_noise_big.jpg") : 
      frame = cv.imread(image_location) 
      # frame = cv.cvtColor(frame,cv.COLOR_RGB2BGR)
      drawer_box(frame)
      plt.imshow(frame,cmap="gray")
      plt.axis('off')
      plt.show()
      cv.waitKey(0)
    else : 
      print("No file found")
  except : 
    print("No face detection")
if __name__ == "__main__":
  main()