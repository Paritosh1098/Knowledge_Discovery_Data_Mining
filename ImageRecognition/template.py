from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import keras as keras
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pylab as pl
from keras.utils import plot_model

# Load the data
labels = np.load('/Users/paritoshgoel/Desktop/ruiz/project3/mylabel.npy')
images = np.load('/Users/paritoshgoel/Desktop/ruiz/project3/images.npy')



# ============= Preprocess data =============#

#Reshaping input data
images_vector = np.reshape(images , (6500,784))
labels_vector = keras.utils.to_categorical(labels, num_classes=10, dtype='int32')

# ============ Stratified Sampling ==============

seed = 7
np.random.seed(seed)
X_fit, X_test, Y_fit, Y_test = train_test_split(images_vector, labels_vector, test_size=0.25, stratify=labels_vector, random_state=seed)
X_train,X_validate, Y_train, Y_validate = train_test_split(X_fit, Y_fit, test_size=0.20, stratify=Y_fit, random_state=seed)
# =============Preprocessing Done ===============#

# Model Template

model = Sequential() # declare model
model.add(Dense(125, input_shape=(28*28,), kernel_initializer='he_normal')) # first layer
model.add(Activation('relu'))

#
#
#
# Fill in Model Here
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
                    validation_data = (X_validate, Y_validate),
                    epochs=50,
                    batch_size=2000)


# Report Results

#print(history.history)

#predicted_classes = model.predict(X_validate)
# Check which items we got right / wrong
#correct_indices = np.nonzero(predicted_classes != Y_validate)[0]
#incorrect_indices = np.nonzero(predicted_classes != Y_test)[0]
#print(Y_validate.shape)
#print(predicted_classes.shape)
score = model.evaluate(X_validate, Y_validate)
print('Test score:', score[0])
print('Test accuracy:', score[1])
predicted_classes = model.predict(X_test)
#plot_model(model, to_file='/Users/paritoshgoel/Desktop/ruiz/Kowledge_Discovery_Data_Mining/project3/model.png')
model.save('/Users/paritoshgoel/Desktop/ruiz/Kowledge_Discovery_Data_Mining/project3/trained_model.proj3');
#Confusion Matrix
actual = [np.argmax(y, axis=None, out=None) for y in Y_test]
predicted = [np.argmax(y, axis=None, out=None) for y in predicted_classes]
cm = confusion_matrix(actual, predicted)

pl.matshow(cm)
pl.title('Confusion matrix')
pl.xlabel('Predicted')
pl.ylabel('True')
pl.colorbar()
pl.show()


#Plot the image
correct_indices = np.nonzero(predicted_classes == Y_test)[0]
incorrect_indices = np.nonzero(predicted_classes != Y_test)[0]

pl.plot(history.history['acc'])
pl.plot(history.history['val_acc'])
pl.title('model accuracy')
pl.ylabel('accuracy')
pl.xlabel('epoch')
pl.legend(['train', 'validation'], loc='upper left')
pl.show()

pl.figure()
for i, incorrect in enumerate(incorrect_indices[10:13]):
    pl.subplot(3,3,i+1)
    pl.imshow(X_test[incorrect].reshape(28,28), cmap='gray', interpolation='none')
    pl.title("Predicted {}, Class {}".format(predicted_classes[incorrect], Y_test[incorrect]))
