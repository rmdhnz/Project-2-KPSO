import cv2 as cv
face_ref = cv.CascadeClassifier("face_ref.xml")
camera = cv.VideoCapture(0)

def face_detection(frame) : 
  optimized_frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
  faces = face_ref.detectMultiScale(optimized_frame,scaleFactor=1.1)
  return faces

def drawer_box(frame) :  
  for x,y,w,h in face_detection(frame) : 
    cv.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),4)

def close_window() :
  camera.release()
  cv.destroyAllWindows()
  exit()
def main() : 
  try : 
    frame = cv.imread("gaussian.jpg")
    # frame = cv.GaussianBlur(frame,(7,7),5)
    # frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    # frame = random_noise(frame,mode="gaussian",mean=0,var=0.1)
    # frame*=255
    # frame = frame.astype("uint8")
    drawer_box(frame)
    cv.imshow("Face Detection",frame)
    cv.waitKey(0)
  except : 
    print("No face detection")
if __name__ == "__main__":
  main()