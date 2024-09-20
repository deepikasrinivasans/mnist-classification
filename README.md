# EX 3 Convolutional Deep Neural Network for Digit Classification

## AIM

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset
- Digit classification and to verify the response for scanned handwritten images.
- The MNIST dataset is a collection of handwritten digits.
- The task is to classify a given image of a handwritten digit into one of 10 classes representing integer values from 0 to 9, inclusively.
- The dataset has a collection of 60,000 handwrittend digits of size 28 X 28. Here we build a convolutional neural network model that is able to classify to it's appropriate numerical value.

## Neural Network Model
![MODEL](https://github.com/user-attachments/assets/736569e2-739b-4ce2-8b34-b25cb97038b9)

## DESIGN STEPS
### STEP 1: Import tensorflow and preprocessing libraries.
### STEP 2: load the dataset
### STEP 3: Scale the dataset between it's min and max values
### STEP 4: Using one hot encode, encode the categorical values
### STEP 5: Split the data into train and test
### STEP 6: Build the convolutional neural network model
### STEP 7: Train the model with the training data
### STEP 8: Plot the performance plot
### STEP 9: Evaluate the model with the testing data
### STEP 10: Fit the model and predict the single input

## PROGRAM

### Name: DEEPIKA S
### Register Number: 212222230028
```py
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
print("DEEPIKA S")

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
print("DEEPIKA S")

X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)

Name: DEEPIKA S

Register Number: 212222230028

model = keras.Sequential()
model.add(layers.Input(shape=(28,28,1)))
model.add(layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train_scaled ,y_train_onehot, epochs=5,
          batch_size=64,
          validation_data=(X_test_scaled,y_test_onehot))

metrics = pd.DataFrame(model.history.history)

metrics.head()

metrics[['accuracy','val_accuracy']].plot()
print("DEEPIKA S")

metrics[['loss','val_loss']].plot()
print("DEEPIKA S")

x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)

print(confusion_matrix(y_test,x_test_predictions))
print('\nDEEPIKA S')

print('DEEPIKA S\n')
print(classification_report(y_test,x_test_predictions))

img = image.load_img('imagenine.jpeg')

type(img)

img = image.load_img('imagenine.jpeg')
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0

x_single_prediction = np.argmax(
    model.predict(img_28_gray_scaled.reshape(1,28,28,1)),
     axis=1)

print('DEEPIKA S\n')
print(x_single_prediction)


plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')

img_28_gray_inverted = 255.0-img_28_gray
img_28_gray_inverted_scaled = img_28_gray_inverted.numpy()/255.0


x_single_prediction = np.argmax(
    model.predict(img_28_gray_inverted_scaled.reshape(1,28,28,1)),
     axis=1)

print(" DEEPIKA S\n")
print(x_single_prediction)
```

## OUTPUT
### Training Loss, Validation Loss Vs Iteration Plot
![IMAGE 1](https://github.com/user-attachments/assets/c10c96cf-9547-47ea-b535-ed050e345bb0)
![IMAGE 2](https://github.com/user-attachments/assets/f7e5f652-beb4-473f-b135-d9fa593dd879)
### Classification Report
![IMAGE 3](https://github.com/user-attachments/assets/c8af4b04-fd3f-4f4b-b696-1d92530b2022)
### Confusion Matrix
![IMAGE 4](https://github.com/user-attachments/assets/903298db-2bd3-4f76-af50-41eb0df25052)
### New Sample Data Prediction
## Input
![IMAGE 5](https://github.com/user-attachments/assets/02164484-2eaf-498b-b1d0-c2648341ac88)
## Output
![IMAGE 6](https://github.com/user-attachments/assets/e5a1882c-efef-46ac-bfe8-4860f421eca1)
## RESULT
Thus, a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images is written and executed successfully.
