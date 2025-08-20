import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Load trained model
# We previously trained and saved a CNN model for digit recognition.
# Here we load that model so we can use it to make predictions on new images.
model = tf.keras.models.load_model("mnist_cnn_model.h5")

def preprocess_image(img_path):
    # Open the image and convert to grayscale.
    # MNIST digits are grayscale, so this ensures compatibility with the trained model.
    img = Image.open(img_path).convert("L")

    # Resize image to 28x28 pixels.
    # The MNIST dataset digits are 28x28, so we must match the input size.
    img = img.resize((28, 28))

    # Convert to a NumPy array for numerical processing.
    img_array = np.array(img)

    # Invert colors if necessary.
    # MNIST digits are white on black. If the input digits are black on white (common),
    # we invert them to match the data distribution the model learned.
    img_array = 255 - img_array

    # Normalize pixel values.
    # Neural networks work better when inputs are scaled between 0 and 1 instead of 0–255.
    img_array = img_array / 255.0

    # Reshape the data to fit the CNN input shape.
    # Model expects a batch (1), height (28), width (28), and channel (1 for grayscale).
    img_array = img_array.reshape(1, 28, 28, 1)

    return img_array

# Loop through your digit images
# We go through all image files in the "my_digits" folder
# and process each one to feed into the trained CNN model.
for filename in os.listdir("my_digits"):
    if filename.endswith((".png", ".jpg", ".jpeg")):
        path = os.path.join("my_digits", filename)

        # Preprocess each image (resize, normalize, reshape, etc.)
        img_array = preprocess_image(path)

        # Use the trained model to predict probabilities for each digit (0–9).
        prediction = model.predict(img_array)

        # np.argmax gives the index of the highest probability,
        # which corresponds to the digit the model thinks is most likely.
        predicted_digit = np.argmax(prediction)

        # Print the prediction result.
        print(f"{filename} -> Predicted digit: {predicted_digit}")
