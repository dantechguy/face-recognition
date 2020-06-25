from imutils import face_utils
import dlib
import cv2
import numpy as np
 
# Vamos inicializar um detector de faces (HOG) para então
# let's go code an faces detector(HOG) and after detect the 
# landmarks on this detected face

# p = our pre-treined model directory, on my case, it's on the same script's diretory.
p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

cap = cv2.VideoCapture(0)

_, image = cap.read()
height, width, channels = image.shape

video = cv2.VideoWriter('video.avi',cv2.VideoWriter_fourcc(*"MJPG"),30,(width*2,height*2))
 
while True:
  # Getting out image by webcam 
  _, image = cap.read()
  # Converting the image to gray scale
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # Generate blank black background
  height, width, channels = image.shape
  height *= 2
  width *= 2
  blank_image = np.zeros((height,width,3), np.uint8)
      
  # Get faces into webcam's image
  rects = detector(gray, 0)
  
  # For each detected face, find the landmark.
  for (i, rect) in enumerate(rects):
    # Make the prediction and transfom it to numpy array
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)

    # Draw on our image, all the finded cordinate points (x,y) 
    for i in range(len(shape)-1):
      if i not in (16, 21, 26, 30, 35, 41, 47, 59):
        x, y = shape[i]
        x *=2
        y *=2
        nx, ny = shape[i+1]
        nx *= 2
        ny *= 2
        cv2.line(blank_image, (width-x, y), (width-nx, ny), (0, 255, 0))
    
    # additional lines
    for i in ((36, 41), (42, 47), (48, 59), (60, 67), (48, 60), (54, 64)):
      index_1 = i[0]
      index_2 = i[1]
      x, y = shape[index_1]
      x *=2
      y *=2
      nx, ny = shape[index_2]
      nx *= 2
      ny *= 2
      cv2.line(blank_image, (width-x, y), (width-nx, ny), (0, 255, 0))
  
  # Show the image
  cv2.imshow("Output", blank_image)
  video.write(blank_image)
  
  k = cv2.waitKey(5) & 0xFF
  if k == 27:
      break

cv2.destroyAllWindows()
video.release()