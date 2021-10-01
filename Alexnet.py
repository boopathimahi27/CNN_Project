import cv2
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers.normalization import BatchNormalization
import numpy as np
folder='E:\Boopathiraja\Data Set\Dataset_short_breakhis'
files = os.listdir(folder)
print(files)
x, y = [], []
for type in files:
    print(type)
    files = os.listdir(folder+'/' +type)
    #print('Reading images from' + folder+'/'+ix+'/'+ ...)
    for file in files[:]:
        img = cv2.imread( folder + '/'+type+'/'+ file)
        img = cv2.resize(img, (227, 227))
        img=img/255.
        x.append(img)
        y.append(type)
        plt.imshow('image',img)
for x1 in x:       
    cv2.imshow("imgae", x1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2

from sklearn.preprocessing import LabelEncoder
L = LabelEncoder()
y_train = L.fit_transform(y_train)
y_test = L.fit_transform(y_test)
y_val=L.fit_transform(y_val)


# Initializing the CNN
classifier = Sequential()

# Convolution Step 1
classifier.add(Convolution2D(96, 11, strides = (4, 4), padding = 'valid', input_shape=(227, 227, 3), activation = 'relu'))

# Max Pooling Step 1
classifier.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'valid'))
classifier.add(BatchNormalization())

# Convolution Step 2
classifier.add(Convolution2D(256, 11, strides = (1, 1), padding='valid', activation = 'relu'))

# Max Pooling Step 2
classifier.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding='valid'))
classifier.add(BatchNormalization())

# Convolution Step 3
classifier.add(Convolution2D(384, 3, strides = (1, 1), padding='valid', activation = 'relu'))
classifier.add(BatchNormalization())

# Convolution Step 4
classifier.add(Convolution2D(384, 3, strides = (1, 1), padding='valid', activation = 'relu'))
classifier.add(BatchNormalization())

# Convolution Step 5
classifier.add(Convolution2D(256, 3, strides=(1,1), padding='valid', activation = 'relu'))

# Max Pooling Step 3
classifier.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'valid'))
classifier.add(BatchNormalization())

# Flattening Step
classifier.add(Flatten())

# Full Connection Step
classifier.add(Dense(units = 4096, activation = 'relu'))
classifier.add(Dropout(0.4))
classifier.add(BatchNormalization())
classifier.add(Dense(units = 4096, activation = 'relu'))
classifier.add(Dropout(0.4))
classifier.add(BatchNormalization())
classifier.add(Dense(units = 1000, activation = 'relu'))
classifier.add(Dropout(0.2))
classifier.add(BatchNormalization())
classifier.add(Dense(units = 2, activation = 'softmax'))


classifier.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

history=classifier.fit(np.array(X_train),y_train,epochs=50,validation_data=(np.array(X_val),y_val))


test_loss,test_acc=classifier.evaluate(np.array(X_test),y_test,verbose=5)
