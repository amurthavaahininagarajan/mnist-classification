# Convolutional Deep Neural Network for Digit Classification

## AIM

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset
Problem Statement: Handwritten Digit Recognition with Convolutional Neural Networks

Objective: Develop a Convolutional Neural Network (CNN) model to accurately classify handwritten digits (0-9) from the MNIST dataset.

Data: The MNIST dataset, a widely used benchmark for image classification, contains grayscale images of handwritten digits (28x28 pixels). Each image is labeled with the corresponding digit (0-9).

## Neural Network Model
![image](https://github.com/amurthavaahininagarajan/mnist-classification/assets/118679102/d99e90d0-ef52-44e5-801f-c66c2aafbcd0)


## DESIGN STEPS
STEP 1: Import tensorflow and preprocessing libraries.
STEP 2: Download and load the dataset
STEP 3: Scale the dataset between it's min and max values
STEP 4: Using one hot encode, encode the categorical values
STEP 5: Split the data into train and test
STEP 6: Build the convolutional neural network model
STEP 7: Train the model with the training data
STEP 8: Plot the performance plot
STEP 9: Evaluate the model with the testing data
STEP 10: Fit the model and predict the single input


## PROGRAM

### Name:AMURTHA VAAHINI.KN
### Register Number:212222240008
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train.shape

X_test.shape

single_image= X_train[0]

single_image.shape

plt.imshow(single_image,cmap='gray')

y_train.shape

X_train.min()

X_train.max()

X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0

X_train_scaled.min()

X_train_scaled.max()

y_train[0]

y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)

type(y_train_onehot)

y_train_onehot.shape

single_image = X_train[500]
plt.imshow(single_image,cmap='gray')

y_train_onehot[500]

X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)

model = keras.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape = (28,28,1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation = 'relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64,activation = 'relu'))
model.add(layers.Dense(10,activation = 'softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics='accuracy')

model.fit(X_train_scaled ,y_train_onehot, epochs=5,
          batch_size=64,
          validation_data=(X_test_scaled,y_test_onehot))

metrics = pd.DataFrame(model.history.history)

print("AMURTHA VAAHINI.KN")
print("212222240008")
metrics.head()

print("AMURTHA VAAHINI.KN")
print("212222240008")
metrics[['accuracy','val_accuracy']].plot()

print("AMURTHA VAAHINI.KN")
print("212222240008")
metrics[['loss','val_loss']].plot()

print("AMURTHA VAAHINI.KN")
print("212222240008")
x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)

print("AMURTHA VAAHINI.KN")
print("212222240008")
print(confusion_matrix(y_test,x_test_predictions))

print("AMURTHA VAAHINI.KN")
print("212222240008")
print(classification_report(y_test,x_test_predictions))

img = image.load_img('/content/images.png')

type(img)

img = image.load_img('/content/images.png')
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0

x_single_prediction = np.argmax(
    model.predict(img_28_gray_scaled.reshape(1,28,28,1)),
     axis=1)

print(x_single_prediction)

print("AMURTHA VAAHINI.KN")
print("212222240008")
plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')

img_28_gray_inverted = 255.0-img_28_gray
img_28_gray_inverted_scaled = img_28_gray_inverted.numpy()/255.0

x_single_prediction = np.argmax(
    model.predict(img_28_gray_inverted_scaled.reshape(1,28,28,1)),
     axis=1)

print("AMURTHA VAAHINI.KN")
print("212222240008")
print(x_single_prediction)

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot
![Screenshot 2024-03-25 141520](https://github.com/amurthavaahininagarajan/mnist-classification/assets/118679102/02629c82-9bd9-4a06-9a39-0c5231f2b578)
![image](https://github.com/amurthavaahininagarajan/mnist-classification/assets/118679102/ccf328e8-b101-4bfe-8a73-1ff7ec998df3)
![Screenshot 2024-03-25 141810](https://github.com/amurthavaahininagarajan/mnist-classification/assets/118679102/21143026-3024-45c1-8bdd-d7666e03d465)


### Classification Report
![Screenshot 2024-03-25 141916](https://github.com/amurthavaahininagarajan/mnist-classification/assets/118679102/85536047-6d0c-4a65-9285-ae4a668ad080)



### Confusion Matrix
![image](https://github.com/amurthavaahininagarajan/mnist-classification/assets/118679102/f54fd213-5e35-420f-89e5-e2603f8740b1)


### New Sample Data Prediction
# Input:
![DL3](https://github.com/amurthavaahininagarajan/mnist-classification/assets/118679102/247a8e66-7669-47e4-80ff-7a22aab23ccb)
# OUTPUT:
![image](https://github.com/amurthavaahininagarajan/mnist-classification/assets/118679102/600756ee-1fd4-4f47-a839-8ca5ff93672c)



## RESULT
Thus a convolutional deep neural network for digit classification is developed and the response for scanned handwritten images is verified.
