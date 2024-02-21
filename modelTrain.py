# this converts a vector class label into matrix important for deep learning as it takes matrix as an input
import random
import string
import cv2

from keras.utils import to_categorical

from keras.optimizers import Adam
from keras.applications import VGG16

from keras.models import load_model, Model

# single line 1D vector [0,2,1,0,3]
# Output ( hot coded matrix) :
# [[1. 0. 0. 0.]
#  [0. 0. 1. 0.]
#  [0. 1. 0. 0.]
#  [1. 0. 0. 0.]
#  [0. 0. 0. 1.]]

#importing tensor flow
import tensorflow as tf

# importing sequential api to create a sequential model
from keras.models import Sequential

# importing layers to add into our sequential model and to create complex neural networks
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, BatchNormalization,Input,concatenate

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import TensorBoard
import os

from keras.callbacks import EarlyStopping,ReduceLROnPlateau , ModelCheckpoint
from keras.utils import plot_model

# Visualization
import matplotlib
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.manifold import TSNE

#data handling
import pandas as pd
import numpy as np
import imutils
import glob #use to retrieve files/pattern matching specific pattern
from sklearn.model_selection import train_test_split


# variables set used in this project
trainDir = "output/train"
valDir = "output/val"
testDir = "output/test"
batchSize = 64
dataDir = 'data/'
targetSize = (64,64)
imgClass = 29
imgChannel = 3


# Labels  generation
labels = []
alphabet = list(string.ascii_uppercase)
labels.extend(alphabet)
labels.extend(["del","nothing","space"])
print(labels)



print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


def samples(labels):
    y_size = 12
    if(len(labels)<10):
        y_size = y_size * len(labels) / 10
    fig,axs = plt.subplots(len(labels), 9, figsize=(y_size, 13))

    for i , label in enumerate(labels):
        axs[i, 0].text(0.5, 0.5, label, ha='center', va='center', fontsize=8)
        axs[i, 0].axis('off')

        label_path = os.path.join(trainDir, label)
        list_files = os.listdir(label_path)

        for j in range(8):
            img_label = cv2.imread(os.path.join(label_path, list_files[j]))
            img_label = cv2.cvtColor(img_label, cv2.COLOR_BGR2RGB)
            axs[i, j + 1].imshow(img_label)
            axs[i, j + 1].axis("off")

    # Title
    plt.suptitle("Sample Images in ASL Alphabet Dataset", x=0.55, y=0.92)

    # Show
    plt.show()

samples(labels)

# Create Metadata for images
list_path = []
list_labels = []
for label in labels:
    label_path = os.path.join(trainDir, label, "*")
    image_files = glob.glob(label_path)

    sign_label = [label] * len(image_files)

    list_path.extend(image_files)
    list_labels.extend(sign_label)

metadata = pd.DataFrame({
    "image_path": list_path,
    "label": list_labels
})

print(metadata)

# Split Dataset to Train 0.7, Val 0.15, and Test 0.15
X_train, X_test, y_train, y_test = train_test_split(
    metadata["image_path"], metadata["label"],
    test_size=0.15,
    random_state=2023,
    shuffle=True,
    stratify=metadata["label"]
)
data_train = pd.DataFrame({
    "image_path": X_train,
    "label": y_train
})

X_train, X_val, y_train, y_val = train_test_split(
    data_train["image_path"], data_train["label"],
    test_size=0.15/0.70,
    random_state=2023,
    shuffle=True,
    stratify=data_train["label"]
)
data_train = pd.DataFrame({
    "image_path": X_train,
    "label": y_train
})
data_val = pd.DataFrame({
    "image_path": X_val,
    "label": y_val
})
data_test = pd.DataFrame({
    "image_path": X_test,
    "label": y_test
})

print(data_train)
print(data_val)
print(data_test)


#data augmentation
# takes data on the fly and generate its variant with different parameters here rescale then those data is fed from the directory
def data_aug():
    train_generator = ImageDataGenerator(rescale=1. / 255).flow_from_dataframe(
        data_train,
        directory='./',
        target_size=targetSize,
        x_col="image_path",
        y_col="label",
        batch_size=batchSize,
        class_mode='categorical',
        # color_mode='grayscale',
    )

    validation_generator = ImageDataGenerator(rescale=1. / 255).flow_from_dataframe(
        data_val,
        directory='./',
        x_col="image_path",
        y_col="label",
        target_size=targetSize,
        batch_size=batchSize,
        class_mode='categorical',
        # color_mode='grayscale',
    )

    test_generator = ImageDataGenerator(rescale=1. / 255).flow_from_dataframe(
        data_test,
        directory='./',
        x_col="image_path",
        y_col="label",
        target_size=targetSize,
        batch_size=batchSize,
        class_mode='categorical',
        # color_mode='grayscale',
        shuffle=False,
    )
    return train_generator , validation_generator , test_generator



def seed_set(seed : int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

seed_set(2023)

train_generator , validation_generator , test_generator = data_aug()



# model = Sequential()
#
# # convolutional  layer adding
#
# model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu',padding='same', input_shape=(200,200,1)))
# model.add(Conv2D(32,3,activation='relu',padding='same'))
# model.add(MaxPooling2D(padding='same'))
# model.add(Dropout(0.2))
#
# model.add(Conv2D(64,3,activation='relu',padding='same'))
# model.add(Conv2D(64,3,activation='relu',padding='same'))
# model.add(MaxPooling2D(padding='same'))
# model.add(Dropout(0.3))
#
#
# model.add(Conv2D(128,3,activation='relu',padding='same'))
# model.add(Conv2D(128,3,activation='relu',padding='same'))
# model.add(MaxPooling2D(padding='same'))
# model.add(Dropout(0.4))
#
#
# #  dense layer adding
# model.add(Flatten())
#
# model.add(Dense(512,activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(128,activation='relu'))
# model.add(Dropout(0.3))
#
#
# # output layer
# model.add(Dense(29, activation='softmax'))

base_model = VGG16(weights="imagenet" , include_top=False, input_shape=(64,64,3))

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = Flatten()(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(29, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.summary()

# earlyStop = EarlyStopping(monitor='val_loss', patience=10)
earlyStop = EarlyStopping(monitor='val_loss',
                              min_delta=0.001,
                              patience= 5,
                              restore_best_weights= True,
                              verbose = 0)

reduceLearn = ReduceLROnPlateau(monitor='val_accuracy',
                                         patience = 2,
                                         factor=0.5 ,
                                         verbose = 1)


model.compile(optimizer='adam', loss = 'categorical_crossentropy' , metrics=['accuracy'])

checkpoint = ModelCheckpoint('best_weight.h5', save_best_only=True, monitor='val_accuracy', mode='max')

# !rm -rf Logs
logdir = os.path.join("Logs")
tensorboard_callback = TensorBoard(log_dir=logdir)

# fitting model
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batchSize,
    epochs=10,
    validation_data=validation_generator,
    verbose=2,
    validation_steps=validation_generator.samples // batchSize,
    callbacks=[ earlyStop , reduceLearn , checkpoint],
)

#saving model
modelJson = model.to_json()
with open("modelVersion2.0.3.json",'w') as json_file:
    json_file.write(modelJson)
model.save("modelVersion2.0.3.h5")


#evaluating
loss,acc = model.evaluate(train_generator , verbose = 0)

scores = model.evaluate(test_generator)
print("%s: %.2f%%" % ("Evaluate Test Accuracy", scores[1]*100))


