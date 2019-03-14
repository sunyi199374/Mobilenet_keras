from keras.models import load_model
import cv2
import numpy as np

model = load_model('mobilenet_parts_model.h5')

img = cv2.imread('test.jpg')
img = cv2.resize(img,(256,256))
img = np.reshape(img,[1,256,256,3])

classes = model.predict_proba(img)

print (classes)