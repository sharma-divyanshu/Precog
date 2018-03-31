import os
import cv2
import numpy as np
from keras.preprocessing import image
from keras.models import load_model

def detect_faces(location, filename):
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    img = cv2.imread(location)
    process_img = np.copy(img)
    gray = cv2.cvtColor(process_img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(process_img, (x,y), (x+w, y+h), (255,255,0), 1)
    processed_images = "static/images/processed/"
    if not os.path.isdir(processed_images):
        os.mkdir(processed_images)
    cv2.imwrite(processed_images + filename, process_img)
    return len(faces), processed_images + filename

def isModi(location, filename):
    classifier = load_model('classifiers/modi.h5')
    test_image = image.load_img(location, target_size = (150,150))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = classifier.predict(test_image)
    if result[0,0] == 1:
        return 'Yes'
    else:
        return 'No'

def isKejriwal(location, filename):
    classifier = load_model('classifiers/kejriwal.h5')
    test_image = image.load_img(location, target_size = (150,150))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = classifier.predict(test_image)
    if result[0,0] == 1:
        return 'Yes'
    else:
        return 'No'