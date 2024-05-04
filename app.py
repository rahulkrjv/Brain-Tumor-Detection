import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from ttkthemes import ThemedStyle

class TumorDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Tumor Detection App")

        self.image_path = None  # Initialize image_path attribute

        # Load the pre-trained model
        self.model = tf.keras.models.load_model('TumorDetectNet.h5')

        # Themed style
        self.style = ThemedStyle(root)
        self.style.set_theme("plastik")  # You can experiment with different themes

        # UI components
        self.label = ttk.Label(root, text="Tumor Detection App", font=("Helvetica", 20, "bold"), foreground="#2c3e50")
        self.label.pack(pady=20)

        self.image_label = ttk.Label(root)
        self.image_label.pack()

        self.browse_button = ttk.Button(root, text="Browse Image", command=self.browse_image)
        self.browse_button.pack(pady=10)

        self.predict_button = ttk.Button(root, text="Predict", command=self.predict_image)
        self.predict_button.pack(pady=10)

        self.result_label = ttk.Label(root, text="", font=("Helvetica", 14), foreground="#27ae60")
        self.result_label.pack(pady=20)

    def browse_image(self):
        file_path = filedialog.askopenfilename(initialdir="/", title="Select Image",
                                            filetypes=(("JPEG files", "*.jpg"), ("all files", "*.*")))
        if file_path:
            self.image_path = file_path  # Set the image path
            self.display_image(file_path)
        else:
            self.result_label.config(text="Error: No image selected.", foreground="#e74c3c")


    def display_image(self, file_path):
        img = Image.open(file_path)
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        self.image_label.config(image=img)
        self.image_label.image = img

        # Save the file path for later use
        self.image_path = file_path

    def predict_image(self):
        try:
            img = tf.keras.preprocessing.image.load_img(
                self.image_path, target_size=(IMG_SIZE, IMG_SIZE))
            img = tf.keras.preprocessing.image.img_to_array(img)
            input_image = np.expand_dims(img, axis=0)

            predictions = self.model.predict(input_image)

            # Get the predicted class index
            predicted_class_index = np.argmax(predictions)

            # Manually define class names
            class_names = ['pituitary', 'notumor', 'meningioma', 'giloma']

            # Get the predicted class label
            predicted_class = class_names[predicted_class_index]
            confidence = np.max(predictions)

            result_text = f"Predicted Class: {predicted_class}, Confidence: {confidence:.2%}"
            self.result_label.config(text=result_text, foreground="#27ae60")

        except Exception as e:
            error_text = f"Error: {e}"
            self.result_label.config(text=error_text, foreground="#e74c3c")


# Set the image size consistent with the model
IMG_SIZE = 256

# Create the Tkinter window
root = tk.Tk()

# Set window size and position
window_width = 800
window_height = 600
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x = (screen_width - window_width) // 2
y = (screen_height - window_height) // 2
root.geometry(f"{window_width}x{window_height}+{x}+{y}")

# Instantiate the TumorDetectionApp class
app = TumorDetectionApp(root)

# Start the Tkinter event loop
root.mainloop()
