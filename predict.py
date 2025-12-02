import tensorflow as tf
import numpy as np
from keras_preprocessing import image

# Load trained model
model = tf.keras.models.load_model("cat_dog_model_2.h5")

# Load image
img = image.load_img(
    './single_prediction/prediction.png',
    target_size=(150, 150)
)

# Convert image to array
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0   # Same scaling as training

# Predict
result = model.predict(img_array)

# Mapping: 0 = cats, 1 = dogs
if result[0][0] > 0.5:
    print("dog")
else:
    print("cat")
