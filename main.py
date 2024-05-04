import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('TumorDetectNet.h5')
IMG_SIZE = 256

# img_path = input("Enter image path:\t")
img_path = r"Testing/meningioma/Te-me_0304.jpg"

# Load and preprocess the input image
try:
    img = tf.keras.preprocessing.image.load_img(
        img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img = tf.keras.preprocessing.image.img_to_array(img)
    input_image = np.expand_dims(img, axis=0)  # Add a batch dimension

    predictions = model.predict(input_image)

    # Get class labels from the model
    class_labels = list(model.layers[-1].get_config()['class_names'])
    label = {0: 'pituitary', 1: 'notumor', 2: 'meningioma', 3: 'giloma'}

    # Find the predicted class
    predicted_class = label[class_labels[np.argmax(predictions)]]

    confidence = np.max(predictions)
    print(f"Predicted Class: {predicted_class}, Confidence: {confidence:.2%}")

except Exception as e:
    print(f"Error: {e}")
