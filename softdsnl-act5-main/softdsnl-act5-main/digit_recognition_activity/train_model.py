import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# Load MNIST dataset (handwritten digits 0–9)
# MNIST is split into training data (x_train, y_train) and test data (x_test, y_test).
# Training data is used to "teach" the model, while test data is used to check how well it learned.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize data
# Pixel values range from 0–255 (grayscale). Dividing by 255 scales them into 0–1.
# Normalization makes training faster and more stable since neural networks work best with small input ranges.
x_train = x_train / 255.0
x_test = x_test / 255.0

# Reshape for CNN
# MNIST images are 28x28 pixels, but CNN layers expect input with an extra "channel" dimension.
# (28, 28, 1) means: height=28, width=28, channels=1 (since grayscale has 1 channel, unlike RGB which has 3).
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Build CNN model
# A CNN (Convolutional Neural Network) is great for image recognition.
# It automatically learns to detect edges, shapes, and patterns from images.
model = models.Sequential([
    # First convolution layer: detects small patterns like edges and corners.
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    # Pooling layer: reduces image size, keeping important features but making training faster.
    layers.MaxPooling2D((2,2)),

    # Second convolution: detects more complex patterns by combining earlier features.
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    # Flatten: turns the 2D image data into a 1D vector so it can be fed into fully connected layers.
    layers.Flatten(),
    # Dense layer: learns higher-level combinations of features (like loops, shapes).
    layers.Dense(64, activation='relu'),
    # Output layer: 10 neurons (one for each digit 0–9), softmax makes outputs into probabilities.
    layers.Dense(10, activation='softmax')
])

# Compile model
# Compilation defines how the model learns:
# - optimizer: "adam" adjusts learning automatically
# - loss: "sparse_categorical_crossentropy" is used for multi-class classification (digits 0–9)
# - metrics: "accuracy" tracks how often predictions are correct
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model
# This is where learning happens. The model sees training images (x_train) and their labels (y_train).
# It adjusts its internal weights to reduce the error. Epochs = number of full passes through the data.
# Validation data checks progress on unseen data (x_test, y_test).
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Evaluate
# After training, we test the model on completely unseen data (x_test, y_test).
# This shows how well the model generalizes to new inputs.
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\nTest accuracy: {test_acc:.4f}")

# Save model
# Saving lets us reuse the trained model later without retraining from scratch.
model.save("mnist_cnn_model.h5")
print("Model saved as mnist_cnn_model.h5")