# Import necessary libraries
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv('brain_tumor_dataset.csv')
IMG_SIZE = 256

# Load and preprocess images
images = []
for image_path in data['image_path']:
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img = tf.keras.preprocessing.image.img_to_array(img)
    images.append(img)

# Convert images to NumPy array
X = np.array(images)
y = data['label'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize a simple CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')  # Assuming 4 classes
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model on the training data
model.fit(X_train, y_train, epochs=10)

# Evaluate the model's accuracy
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Accuracy: {test_accuracy * 100:.2f}%')

# Save the model to a file
model.save('TumorDetectNet.h5')
