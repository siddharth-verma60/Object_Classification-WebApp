import numpy as np
import pandas as pd
import os
from keras.preprocessing.image import img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Defining the dimensions of input images
INPUT_HEIGHT = 64
INPUT_WIDTH = 64
INPUT_CHANNELS =3

# Loading the dataset
dataset_directory = os.getcwd() + "/dataset"
filenames = os.listdir(dataset_directory)
categories = []
for filename in filenames:
    category = filename.split('.')[0]
    if category == 'dog':
        categories.append((0,1))
    elif category == 'cat':
        categories.append((1,0))
    else:
        categories.append((0,0))

df = pd.DataFrame({
    'filename': filenames,
    'label': categories
})

images = []
for i in tqdm(range(df.shape[0])):
    img = load_img(dataset_directory+"/"+df['filename'][i],target_size=(INPUT_WIDTH,INPUT_HEIGHT,INPUT_CHANNELS))
    img = img_to_array(img)
    img = img/255
    images.append(img)
X = np.array(images)

labels = []
for i in range(df.shape[0]):
    label = np.array(df['label'][i])
    labels.append(label)
Y=np.array(labels)

# Splitting into training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42, test_size=0.2)

def create_cnn_model():

    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(INPUT_WIDTH, INPUT_HEIGHT, INPUT_CHANNELS)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # One class each for dog and cat
    # Sigmoid is used instead of softmax because each class is independant of the other.
    model.add(Dense(2, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()

    return model

# Create the computation graph
CNN_model = create_cnn_model()

# Start the training
History = CNN_model.fit(X_train, Y_train, epochs=20, validation_data=(X_test, Y_test), batch_size=64)

#Save the model
# serialize model to JSON
model_json = CNN_model.to_json()
with open(os.getcwd() + "/model/model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
CNN_model.save_weights(os.getcwd() + "/model/model.h5")