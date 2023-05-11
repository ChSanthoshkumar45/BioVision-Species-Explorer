import tensorflow as tf
import numpy as np
import cv2

# Load the trained model
model = tf.keras.models.load_model('trained_model.h5')

# Define the class labels
class_labels = ['African_Elephant',
 'Amur_Leopard',
 'Arctic_Fox',
 'Chimpanzee',
 'Jaguars',
 'Lion',
 'Orangutan',
 'Panda',
 'Panthers',
 'Rhino',
 'cheetahs']

# Load and preprocess the image
img = cv2.imread(r"C:\Users\sn902\Python Programming\Danger Of Extinction\Panda\1580806672147_630.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (150, 150,))
img = np.expand_dims(img, axis=0)
img = img / 255.0

# Make a prediction using the model
predictions = model.predict(img)

# Get the predicted class label
predicted_label = class_labels[np.argmax(predictions)]

# Print the predicted label
print('The predicted label is:', predicted_label)