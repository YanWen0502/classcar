from tensorflow.keras.models import load_model
import cv2
import time
import numpy as np

model = load_model('cnn.h5')
model.load_weights('cnn.h5')
model.summary()
cap = cv2.VideoCapture(0)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

while(True):
    ret, frame =cap.read()

    if ret == True:
        cv2.imshow('Camera', frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        gray_INV = 255-gray
        
        standard = gray_INV - np.amin(gray_INV)
        
        kernel = np.ones((5,5), np.uint8)
        img_dilate = cv2.dilate(standard, kernel, iterations = 3)
        
        img_resize = cv2.resize(img_dilate, (28,28))
        
        cv2.imshow('input', img_resize)
        
        normalization = img_resize / (np.amax(img_resize))
        
        x_image = np.reshape(normalization, (1,28,28,1)).astype('float32')

