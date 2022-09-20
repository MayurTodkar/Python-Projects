import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import tensorflow as tf
from tensorflow.keras import Sequential
# from keras.layers import Dense,Dropout,Activation,Flatten
import pickle
from tensorflow.python.keras.engine.sequential import relax_input_shape

from tensorflow.python.keras.layers.convolutional import Conv2D, Conv2DTranspose
from tensorflow.python.keras.layers.core import Activation, Dense, Flatten
from tensorflow.python.keras.layers.pooling import MaxPooling2D

DataDir= r"D:\Project\Datasets\Cow_images/"

CATEGORIES=["Cows_indigenous", "Cows_jersy"]

for i in CATEGORIES:
    path=os.path.join(DataDir,i)
    for img in os.listdir(path):
        img_array=cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_array,cmap='gray')
        plt.show()
        break
    break


img_size=100

new_array=cv2.resize(img_array,(img_size,img_size))
plt.imshow(new_array,cmap='gray')
plt.show()

training_data=[]

def create_training_data():
    for i in CATEGORIES:

        path=os.path.join(DataDir,i)
        class_num=CATEGORIES.index(i)

        for img in os.listdir(path):
            try:
                img_array=cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                new_array=cv2.resize(img_array,(img_size,img_size))
                training_data.append([new_array,class_num])

            except Exception as e:
                pass

create_training_data()
print(len(training_data))


import random
random.shuffle(training_data)
for sample in training_data[:10]:
    print(sample)



X=[]
y=[]

for features,label in training_data:
    X.append(features)
    y.append(label)

print(X[0].reshape(-1,img_size,img_size,1))


X=np.array(X).reshape(-1,img_size,img_size,1)


import pickle

pickle_out=open(r"D:\Project\X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out=open(r"D:\Project\y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()



# from keras.layers import Conv2D, MaxPooling2D

pickle_in=open(r"D:\Project\X.pickle","rb")
X=pickle.load(pickle_in)

pickle_in=open(r"D:\Project\y.pickle","rb")
y=pickle.load(pickle_in)

print(X)

y=np.array(y)
X=X/255.0

print(X)

model= Sequential()

model.add(Conv2D(256,(3,3),input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Conv2D(256,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))


model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(X,y,batch_size=4,epochs=10,validation_split=0.3)

model.save(r"D:\Project\indigenousVSjersy_CNN.model")