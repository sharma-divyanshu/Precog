from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.models import load_model, model_from_yaml
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as kb

training_set = '../dataset/modi/train'
test_set = '../dataset/modi/test'

epochs = 35
batch_size = 64

train_samples = 1000
w, h = 150, 150
test_samples = 300

if kb.image_data_format() == 'channels_first':
    input_shape = (3, w, h)
else:
    input_shape = (w, h, 3)

classifier = Sequential()

classifier.add(Conv2D(128, (3, 3), input_shape=(h, w, 3)))
classifier.add(Activation('relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Conv2D(64, (3, 3)))
classifier.add(Activation('relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Conv2D(32, (3, 3)))
classifier.add(Activation('relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Conv2D(32, (3, 3)))
classifier.add(Activation('relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Flatten())
classifier.add(Dense(32))
classifier.add(Activation('relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(16))
classifier.add(Activation('relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(1))
classifier.add(Activation('sigmoid'))

classifier.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, horizontal_flip=True, zoom_range=0.2)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(training_set,
                                                    target_size=(w, h),
                                                    batch_size=batch_size,
                                                    class_mode='binary')

test_generator = test_datagen.flow_from_directory(test_set,
                                                  target_size=(w, h),
                                                  batch_size=batch_size,
                                                  class_mode='binary')

classifier.fit_generator(train_generator,
                         steps_per_epoch=64,
                         epochs=epochs,
                         validation_data=test_generator,
                         validation_steps=test_samples // batch_size)



classifier.save('../classifiers/modi_wo.h5', include_optimizer=False)
#
#import numpy as np
#from keras.preprocessing import image
#
#test_image = image.load_img('test3.jpg', target_size = (150,150))
#test_image = image.img_to_array(test_image)
#test_image = np.expand_dims(test_image, axis = 0)
#result = classifier.predict(test_image)
#print(result)
