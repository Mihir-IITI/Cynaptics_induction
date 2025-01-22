from keras.utils import to_categorical
from keras_preprocessing.image import load_img
from keras.models import Sequential
from keras.applications import MobileNetV2, ResNet152, VGG16, EfficientNetB0, InceptionV3
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
import os #used to access the directory where files are stored
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm #shows progress bar

def createdataframe(dir):
    image_paths = []
    labels = []
    for label in os.listdir(dir):
        for imagename in os.listdir(os.path.join(dir, label)):
            image_paths.append(os.path.join(dir, label, imagename))
            labels.append(label)
        print(label, "completed")
    return image_paths, labels

#converts images to array
#note in test data all ai image are .png and all real image are .jpg; will nn learn to distinguish between the formats(evn tho it is converted to array)
def extract_features(images):
    features = []
    for image in tqdm(images):
        img = load_img(image, target_size=(236, 236)) #resizes image to 236X236 pixel
        img = np.array(img) #converts images into numpy array
        features.append(img) #saves the numpy array inside features array
    features = np.array(features)
    features = features.reshape(features.shape[0], 236, 236, 3)  # Reshape all images in one go
    return features

TRAIN_DIR = "/content/drive/MyDrive/kaggle/datasets/induction-task/Data/Train" # Corrected path


train = pd.DataFrame() #creates 2 dimensional, mutable, labeled array, here the labels(column names) are 'image', 'label'
train['image'], train['label'] = createdataframe(TRAIN_DIR) #train['train'] stores the file location of image, train['label'] stores name of files

train_features = extract_features(train['image']) #scales images to predetermined size; converts it into array; saves them as train_features[]

x_train = train_features / 255.0

#train['label'] stores labels in form of strings, this block of code creates an array y_train which stores integers as labels
le = LabelEncoder()
le.fit(train['label'])
y_train = le.transform(train['label'])
y_train = to_categorical(y_train, num_classes=2)

#from here on only x_train and y_train will be used
#all the above code converted image data into a form the neural network can use
#the neural network starts from here

model = Sequential() #neural network object; Sequential(belongs to keras)

# Convolutional layers
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(236, 236, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(1024, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(2048, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x=x_train, y=y_train, batch_size=25, epochs=5) #how does this work?

#added by me
TEST_DIR = "/content/drive/MyDrive/kaggle/datasets/induction-task/Data/Test" # Corrected path
test = pd.DataFrame()
test['image'] = [os.path.join(TEST_DIR, imagename) for imagename in os.listdir(TEST_DIR)]
test_features = extract_features(test['image'])
x_test = test_features / 255.0
y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
y_pred = le.inverse_transform(y_pred)
test['label'] = y_pred
test.to_csv('submission.csv', index=False)

