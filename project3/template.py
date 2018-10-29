from keras.models import Sequential
from keras.layers import Dense, Activation
import keras as keras
import numpy as np

# Load the data
labels = np.load('./mylabel.npy')
images = np.load('./images.npy')



# ============= Preprocess data =============#

#Reshaping input data
output_classes = [0,1,2,3,4,5,6,7,8,9]
images_vector = np.reshape(images , (6500,784))
output_classes_x_hot = keras.utils.to_categorical(output_classes, num_classes=10, dtype='int32')

# Create Training set , Validation set , Test set using Stratified sampling

# ============ Stratified Sampling ==============

#initialize map with empty array
mapOflablesWithImages = {k: [] for k in range(10)}
index = 0
print(mapOflablesWithImages[index])

#fill the map
for label in labels:
     mapOflablesWithImages[label].append(images_vector[index])
     index = index + 1

training_images = []
training_lables = []

validation_images = []
validation_lables = []

test_images = []
test_lables = []
noOfDataInstances = {}
for x in range(0,10):
    noOfDataInstances[x] = len(mapOflablesWithImages[x])

    for j in range(0, int (noOfDataInstances[x] * 0.6)):
        training_images.append(mapOflablesWithImages[x][j])
        training_lables.append(x)
    for j in range(int(noOfDataInstances[x] * 0.6) + 1 , int(noOfDataInstances[x] * 0.75)):
        validation_images.append(mapOflablesWithImages[x][j])
        validation_lables.append(x)

    for j in range(int (noOfDataInstances[x] * 0.75) + 1 , noOfDataInstances[x]):
        test_images.append(mapOflablesWithImages[x][j])
        test_lables.append(x)

# ============ Sampling done ! =============#

# shuffle the lists in same order
#training_images, training_lables = (list(t) for t in zip(*sorted(zip(training_images, training_lables))))
#validation_images, validation_lables = (list(t) for t in zip(*sorted(zip(validation_images, validation_lables))))
#test_images, test_lables = (list(t) for t in zip(*sorted(zip(test_images, test_lables))))

# =============Preprocessing Done ===============#
print(training_images[0])


# Model Template

model = Sequential() # declare model
model.add(Dense(10, input_shape=(28*28, ), kernel_initializer='he_normal')) # first layer
model.add(Activation('relu'))
#
#
#
# Fill in Model Here
model.add(Dense(15))
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
history = model.fit(training_images, training_lables,
                    validation_data = (validation_images, validation_lables),
                    epochs=10,
                    batch_size=512)


# Report Results

print(history.history)
model.predict()
