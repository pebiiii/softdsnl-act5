# Activity: Handwritten Digit Recognition with TensorFlow

In this activity, you will train a **Convolutional Neural Network (CNN)** using the **MNIST dataset** (handwritten digits `0–9`).  
After training, you will test the model using **your own handwritten digits** as input.

---

## What Does Training a Model Mean?
When we say **training a model**, we are teaching the computer to **recognize patterns** from data.  
- In this case, the data is images of handwritten digits.  
- During training, the model looks at thousands of digit images and tries to "guess" which digit each image represents.  
- Each time it makes a mistake, it **adjusts itself** slightly so it can improve for the next guess.  
- After many rounds, the model becomes very good at recognizing digits it has never seen before.

Think of it like teaching a child to recognize numbers by showing them flashcards thousands of times until they learn the shapes.

---

## Requirements

Create a `requirements.txt` with the following content:

```
tensorflow
numpy
matplotlib
pillow
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Folder Structure

```
digit_recognition_activity/
│── requirements.txt
│── README.md
│── train_model.py
│── test_model.py
│── my_digits/        # put your own digit images here (e.g., digit0.png, digit1.png)
```

---

## Step 1: Train the Model

Save the following as `train_model.py`:

```python
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
```

Run training:

```bash
python train_model.py
```

---

## Understanding Model Training Results

When you train your MNIST model, you’ll see metrics like **accuracy, loss, val_accuracy, and val_loss**.  
Here’s what they mean:

### Metrics Explained

1. **Accuracy (`accuracy`)**  
   - How many predictions the model got correct during training.  
   - Example: If it sees 1000 images and predicts 920 correctly → **92% accuracy**.  
   - Higher accuracy = better learning.

2. **Loss (`loss`)**  
   - A number showing **how wrong the model is**.  
   - Calculated by a function (like *categorical crossentropy*) that penalizes wrong answers.  
   - Lower loss = better performance.  
   - Think of it as a “penalty score” → smaller is better.

3. **Validation Accuracy (`val_accuracy`)**  
   - Accuracy on **new images the model never saw before** (validation set).  
   - Tells us if the model can generalize, not just memorize.  
   - If training accuracy is 98% but validation accuracy is 85% → the model may be **overfitting**.

4. **Validation Loss (`val_loss`)**  
   - Loss on the validation set.  
   - Important for checking generalization.  
   - If training loss decreases but validation loss increases → also a sign of **overfitting**.

---

### In Simple Terms:
- **Accuracy** = % correct on training data.  
- **Loss** = how wrong the model is on training data.  
- **Validation Accuracy** = % correct on unseen data.  
- **Validation Loss** = how wrong the model is on unseen data.  

Goal: **High accuracy + low loss** on both training and validation.

---

## Step 2: Test with Your Own Handwritten Digits

1. Write digits `0–9` on a white paper.  
2. Take a clear photo and crop into **individual digit images** (PNG/JPG).  
3. Place them in the `my_digits/` folder.  
   - Example: `digit0.png`, `digit1.png`, ...

Save the following as `test_model.py`:

```python
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

```

Run testing:

```bash
python test_model.py
```

---

## Deliverables

- A **link to your GitHub repo fork** containing your code.  
- A **PDF output file** that shows:
  - Screenshots of model training results (accuracy/loss).  
  - Screenshots of predictions on your handwritten digits.  

---
