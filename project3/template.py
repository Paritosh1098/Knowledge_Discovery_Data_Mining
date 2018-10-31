from keras.models import Sequential
from keras.layers import Dense, Activation
import keras as keras
import numpy as np
from sklearn.model_selection import train_test_split

# Load the data
labels = np.load('./mylabel.npy')
images = np.load('./images.npy')



# ============= Preprocess data =============#

#Reshaping input data
images_vector = np.reshape(images , (6500,784))
labels_vector = keras.utils.to_categorical(labels, num_classes=10, dtype='int32')

# ============ Stratified Sampling ==============

seed = 7
np.random.seed(seed)
X_train, X_test, Y_train, Y_test = train_test_split(images_vector, labels_vector, test_size=0.33, random_state=seed)
# =============Preprocessing Done ===============#

# Model Template

model = Sequential() # declare model
model.add(Dense(512, input_shape=(28*28,), kernel_initializer='he_normal')) # first layer
model.add(Activation('relu'))
#
#
#
# Fill in Model Here
model.add(Dense(512))
model.add(Activation('relu'))
#
#



model.add(Dense(10, kernel_initializer='he_normal')) # last layer
model.add(Activation('softmax'))


# Compile Model
model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train Model


history = model.fit(X_train, Y_train,
                    validation_data = (X_test, Y_test),
                    epochs=10,
                    batch_size=2000)


# Report Results

print(history.history[1])
