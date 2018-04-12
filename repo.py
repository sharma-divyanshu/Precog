import os
import cv2
import numpy as np
from keras.preprocessing import image
from keras.models import load_model

import h5py
import tensorflow as tf

def detect_faces(location, filename, app_dir):
    face_cascade = cv2.CascadeClassifier(os.path.join(app_dir, 'haarcascades/haarcascade_frontalface_default.xml'))
    img = cv2.imread(location)
    process_img = np.copy(img)
    gray = cv2.cvtColor(process_img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(process_img, (x,y), (x+w, y+h), (255,255,0), 1)
    processed_images = os.path.join(app_dir, 'static/images/')
    if not os.path.isdir(processed_images):
        os.mkdir(processed_images)
    cv2.imwrite(processed_images + filename, process_img)
    return len(faces), 'static/images/' + str(filename)

def loadClassifier(app_dir):
    classifier_nm = load_model(os.path.join(app_dir, 'classifiers/modi.h5'))
    classifier_nm.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    classifier_ak = load_model(os.path.join(app_dir, 'classifiers/kejriwal.h5'))
    classifier_ak.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    graph = tf.get_default_graph()
    return graph, classifier_nm, classifier_ak
    
def isModi(location, filename, app_dir, classifier_nm):
#    classifier = load_model(os.path.join(app_dir, 'classifiers/modi.h5'))
#    classifier.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    test_image = image.load_img(location, target_size = (150,150))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = classifier_nm.predict(test_image)
    if result[0,0] == 1:
        return 'Yes'
    else:
        return 'No'

def isKejriwal(location, filename, app_dir, classifier_ak):
#    classifier = load_model(os.path.join(app_dir, 'classifiers/kejriwal.h5'))
#    classifier.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    test_image = image.load_img(location, target_size = (150,150))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = classifier_ak.predict(test_image)
    if result[0,0] == 1:
        return 'Yes'
    else:
        return 'No'