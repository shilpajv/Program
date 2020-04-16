# Convolutional Neural Network
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D


# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))



# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))

classifier.add(Dense(units = 4, activation = 'softmax'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('C:/Users/aksha.LAPTOP-KKEAULMJ/OneDrive/Documents/Python/Ineuron/cnn/family',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('C:/Users/aksha.LAPTOP-KKEAULMJ/OneDrive/Documents/Python/Ineuron/cnn/familytest',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'categorical')

model=classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 1,
                         validation_data = test_set,    
                         validation_steps = 1000)

# Part 3 - Making new predictions

classifier.save('model.h5')


import numpy as np
from keras.preprocessing import image
test_image = image.load_img('C:/Users/aksha.LAPTOP-KKEAULMJ/OneDrive/Documents/Python/Ineuron/cnn/test2.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
print("result shape")
print(result.shape)
result = classifier.predict(test_image)
training_set.class_indices
#{'akshay': 0, 'kishor': 1, 'mom': 2, 'shilpa': 3} 


#######Prediction code#####

if result[0][0] == 0:
    prediction = 'akshay'
    print(prediction)
elif result[0][0] == 1:
    prediction = 'kishor'
    print(prediction)
elif result[0][0]==2:
    prediction='mom'
    print (prediction)
else:
    prediction ='shilpa'
    print(prediction)
### not giving correct output







