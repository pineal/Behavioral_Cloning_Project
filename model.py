import csv
import cv2
from sklearn.utils import shuffle

#data selection
lines = []
with open('./data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

images = []
measurements = []
for line in lines:
	for i in range(3):
		source_path = line[i]
		tokens = source_path.split('/')
		filename = tokens[-1]
		local_path = "./data/IMG/" + filename
		image = cv2.imread(local_path)
		images.append(image)
	correction = 0.2
	measurement = float(line[3])
	measurements.append(measurement)
	measurements.append(measurement + correction)
	measurements.append(measurement - correction)

#image augmentation
images, measurements = shuffle(images, measurements)

augmented_images = []
augmented_measurements = []
for image, measurement in zip(images, measurements):
	augmented_images.append(image)
	augmented_measurements.append(measurement)
	flipped_image = cv2.flip(image, 1)
	flipped_measurement = float(measurement) * -1.0
	augmented_images.append(flipped_image)
	augmented_measurements.append(flipped_measurement)


print (len(images))
print (len(measurements))

import numpy as np

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

print (len(X_train))
print (len(y_train))


import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

# Nvidia CNN model
model = Sequential()
# Normalized input planes
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
# Added a layer to crop the image
model.add(Cropping2D(cropping=((50,20), (0,0))))
# Five convolutional layers
model.add(Convolution2D(24,5,5,activation='relu', subsample=(2,2)))
model.add(Convolution2D(36,5,5,activation='relu', subsample=(2,2)))
model.add(Convolution2D(48,5,5,activation='relu', subsample=(2,2)))
model.add(Convolution2D(64,3,3,activation='relu', subsample=(2,2)))
model.add(Convolution2D(64,3,3,activation='relu', subsample=(2,2)))
# Add a dropout layer to reduce overfitting with a keep prob = 0.5
model.add(Dropout(0.5))
model.add(Flatten())
# Fully-connected layers
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

# config the model for training
model.compile(optimizer='adam', loss='mse')

# split 20% of training data for validation, trains the model for a fixed number of 10 epochs
model.fit(X_train, y_train, validation_split=0.2, shuffle=True)

model.save('model.h5')

exit()

