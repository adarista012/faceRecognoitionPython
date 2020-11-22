import cv2
import numpy as np


faceClassif = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

img = cv2.imread('imagen_input.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = faceClassif.detectMultiScale(gray,
scaleFactor=1.1,
minNeighbors=5,
minSize=(30,30),
maxSize=(200,200))

for(x,y,w,h) in faces:
   cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

cv2.imshow('img', img)

cv2.waitKey(0)
cv2.destroyAllWindows()



