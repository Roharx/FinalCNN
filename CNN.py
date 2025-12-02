import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
import numpy as np
from keras_preprocessing import image

# ------------------------------------------------------------------------------------------------------------------
# Data Preprocessing
# ------------------------------------------------------------------------------------------------------------------
# Image Augmentation
training_datagen = ImageDataGenerator(
    rescale = 1./255, # Applies feature scaling to every pixel value (every pixel = 0-255 -> 0-1)
    shear_range = 0.2, # Applies shear/tilt to the image
    zoom_range = 0.2, # Random zoom in/out
    horizontal_flip = True # Random flips
)

# Import training set
training_set = training_datagen.flow_from_directory(
    './training_set',
    target_size = (150, 150), # Final size of the images when they will be fitted into the CNN
    batch_size = 32, # How many images do we want in each batch, 32 is a classic default value
    class_mode = 'binary' # Binary or categorical, in this case, we want to choose between cat or dog so binary
)

# Import test set
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(
    './test_set',
    target_size = (150, 150),
    batch_size = 32,
    class_mode = 'binary'
)

# ------------------------------------------------------------------------------------------------------------------
# Build CNN
# ------------------------------------------------------------------------------------------------------------------
cnn = tf.keras.models.Sequential()

# 1st convolution layer
cnn.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)))

# 2nd layer
cnn.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(2,2))

# 3rd layer
cnn.add(tf.keras.layers.Conv2D(128, (3,3), activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(2,2))

# 4th layer
cnn.add(tf.keras.layers.Conv2D(256, (3,3), activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(2,2))

# Flatten
cnn.add(tf.keras.layers.Flatten())

# Fully connected
cnn.add(tf.keras.layers.Dense(256, activation='relu'))

# Dropout (prevents overfitting)
cnn.add(tf.keras.layers.Dropout(0.5))

# Output layer
cnn.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# ------------------------------------------------------------------------------------------------------------------
# Train CNN
# ------------------------------------------------------------------------------------------------------------------
# Compiling
cnn.compile(
    optimizer = 'adam',
    loss = 'binary_crossentropy',
    metrics = ['accuracy']
)

# Training
cnn.fit(
    x = training_set,
    validation_data = test_set,
    epochs = 25
)

# Save the model
cnn.save("cat_dog_model.h5")

# ------------------------------------------------------------------------------------------------------------------
# Predict
# ------------------------------------------------------------------------------------------------------------------
test_image = image.load_img(
    './single_prediction/prediction.png',
    target_size = (64, 64)
)
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
# print(training_set.class_indices)
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'

print(prediction)