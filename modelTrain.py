# this converts a vector class label into matrix important for deep learning as it takes matrix as an input
from keras.utils import to_categorical
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
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

# importing optimizer to minimise the loss
from keras.optimizers import  Adam

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import TensorBoard
import os


batchSize = 48
dataDir = 'data/'
targetSize = (48,48)


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


# takes data on the fly and generate its variant with different parameters here rescale then those data is fed from the directory
trainGenerator = ImageDataGenerator(samplewise_center=True,
                                    samplewise_std_normalization=True,
                                    validation_split=0.1,
                                    rescale=1./255).flow_from_directory(
    'output/train',
    target_size= targetSize,
    batch_size= batchSize,
    class_mode= 'categorical',
    color_mode= 'grayscale',
    shuffle=True,
    subset="training"
)

validGenerator = ImageDataGenerator(samplewise_center=True,
                                    samplewise_std_normalization=True,
                                    validation_split=0.1,
                                    rescale=1./255).flow_from_directory(
    'output/val',
    target_size=targetSize,
    batch_size= batchSize,
    class_mode= 'categorical',
    color_mode= 'grayscale',
    shuffle=True,
    subset="validation"
)

labels = list(trainGenerator.class_indices.keys())

model = Sequential()

# convolutional  layer adding

model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu', input_shape=(48,48,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(filters=256, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))


model.add(Conv2D(filters=512, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Flatten())

#  dense layer adding

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))

# output layer
model.add(Dense(29, activation='softmax'))

model.summary()

# earlyStop = EarlyStopping(monitor='val_loss', patience=10)

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = 'accuracy' )

# !rm -rf Logs
logdir = os.path.join("Logs")
tensorboard_callback = TensorBoard(log_dir=logdir)

# fitting model
model.fit(
    trainGenerator,
    steps_per_epoch=trainGenerator.samples // batchSize,
    epochs=50,
    validation_data=validGenerator,
    validation_steps=validGenerator.samples // batchSize,
    callbacks=[
            tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=2,
            restore_best_weights=True)]
)

#saving model
modelJson = model.to_json()
with open("modelVersion1.0.1.json",'w') as json_file:
    json_file.write(modelJson)
model.save("modelVersion1.0.1.h5")
