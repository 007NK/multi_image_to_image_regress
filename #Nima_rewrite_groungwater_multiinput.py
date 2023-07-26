import numpy as np
import pandas as pd
import os
import cv2
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Conv2D, Concatenate, Dense, Flatten, Reshape, UpSampling2D, MaxPooling2D

from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

###############################################################################


# Define the paths to the input and output image folders, and the CSV file with the numeric input data
current_directory = os.getcwd()

input_dir_conductivity_lower_layer = current_directory + '/conductivity_lower_layer'
input_dir_conductivity_upper_layer = current_directory + '/conductivity_upper_layer'
input_dir_initial_heads = current_directory + '/initial_heads'
csv_path = current_directory + '/data/recharge.csv'




n_years = 30
stress_periods_per_year = 2
n_stress_periods = n_years * stress_periods_per_year
repetitions = 10

###############################################################################
# Load the input and output images
conductivity_lower_layer = []
conductivity_upper_layer = []
initial_heads = []
target_heads_1 = []
target_heads_2 = []

for i in os.listdir(input_dir_conductivity_lower_layer):
    image = cv2.imread(input_dir_conductivity_lower_layer+f'/{i}', cv2.IMREAD_GRAYSCALE)/255.0
    image = cv2.resize(image, (514, 1108))
    conductivity_lower_layer.append(image)

for i in os.listdir(input_dir_conductivity_upper_layer):
    image = cv2.imread(input_dir_conductivity_upper_layer+f'/{i}', cv2.IMREAD_GRAYSCALE)/255.0
    image = cv2.resize(image, (514, 1108))
    conductivity_upper_layer.append(image)

for i in os.listdir(input_dir_initial_heads):
    image = cv2.imread(input_dir_initial_heads+f'/{i}', cv2.IMREAD_GRAYSCALE)/255.0
    image = cv2.resize(image, (514, 1108))
    initial_heads.append(image)

total_data_length = len(initial_heads)
#OR (initial_heads).shape[0]

conductivity_lower_layer = np.array(conductivity_lower_layer)
conductivity_upper_layer = np.array(conductivity_upper_layer)

initial_heads = np.array(initial_heads)
target_heads_1 = np.array(target_heads_1)
target_heads_2 = np.array(target_heads_2)


# Load the numeric input data from the CSV file
recharge = pd.read_csv(csv_path, header=None).values.astype('float32')
numeric_input = recharge / np.max(recharge) # normalize the numeric input data

input_train, input_test, output_train, output_test, numeric_train, numeric_test = 
train_test_split(input_imgs, output_imgs, numeric_input, test_size=0.2)


##################################################################################

# Define input layers for the 3 input images and the numeric data
input_layer1 = Input(shape=(514, 1108, 1))
input_layer2 = Input(shape=(514, 1108, 1))
input_layer3 = Input(shape=(514, 1108, 1))
input_layer4 = Input(shape=(1,))

# Define CNN layers for the 3 parallel networks
conv1_1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer1)
conv2_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1_1)
pool1_1 = MaxPooling2D(pool_size=(2, 2))(conv2_1)
conv3_1 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1_1)

conv1_2 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer2)
conv2_2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1_2)
pool1_2 = MaxPooling2D(pool_size=(2, 2))(conv2_2)
conv3_2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1_2)

conv1_3 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer3)
conv2_3 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1_3)
pool1_3 = MaxPooling2D(pool_size=(2, 2))(conv2_3)
conv3_3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1_3)

# Define fully connected layers for the numeric input data
dense1 = Dense(64, activation='relu')(input_layer4)
dense2 = Dense(128, activation='relu')(dense1)
flatten1 = Flatten()(dense2)

# Concatenate the output features of the 3 parallel networks and the fully connected layers
concatenate_layer = Concatenate()([conv3_1, conv3_2, conv3_3, flatten1])

# Define CNN layers for the final output network
up1 = UpSampling2D(size=(2, 2))(concatenate_layer)
conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1)
up2 = UpSampling2D(size=(2, 2))(conv4)
conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(up2)
output_layer = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(conv5)

# Define the model
model = Model(inputs=[input_layer1, input_layer2, input_layer3, input_layer4], outputs=output_layer)

# Compile the model
model.compile(optimizer=Adam(lr=1e-4), loss='mse')

####################################################################################################train
# Define the batch size and number of epochs

batch_size = 2
num_epochs = 10


# Train the model
history = model.fit([train_input1, train_input2, train_input3, train_numeric_input], train_output,
          epochs=num_epochs,
          batch_size=batch_size,
          verbose=1, validation_split=0.2,
          validation_data=([val_input1, val_input2, val_input3, val_numeric_input], val_output))


#history = model.fit([input_train[:, :, :, 0].reshape(-1, 514, 1108, 1), input_train[:, :, :, 1].reshape(-1, 514, 1108, 1), input_train[:, :, :, 2].reshape(-1, 514, 1108, 1), numeric_train], [output_train[:, :, :, 0].reshape(-1, 514, 1108, 1), output_train[:, :, :, 1].reshape(-1, 514, 1108, 1)], batch_size=batch_size, epochs=num_epochs, verbose=1, validation_split=0.2)



# Plot the percentage error vs epoch number for the training and validation datasets
train_loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(train_loss)
plt.plot(val_loss)
plt.title('Model Loss')
plt.ylabel('Percentage Error')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.savefig('loss.png')
plt.clf()