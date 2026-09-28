import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
import ssl
import random

# Bypass SSL certificate verification for dataset download (Mac specific)
ssl._create_default_https_context = ssl._create_unverified_context


def run_mnist_cnn():
    print("1. Loading MNIST Dataset...")
    # Load the handwritten digits dataset
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

    print("2. Preprocessing Data...")
    # Scale pixel values from 0-255 down to 0.0-1.0 for the neural network
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    # CNNs expect a 4D tensor: (batch_size, height, width, color_channels)
    # We add a 1 for the color channel because MNIST is grayscale
    X_train = np.expand_dims(X_train, -1)
    X_test = np.expand_dims(X_test, -1)

    print("3. Building CNN Architecture...")
    # Sequential groups a linear stack of layers
    model = models.Sequential([
        # Dedicated Input layer for modern Keras
        layers.Input(shape=(28, 28, 1)),

        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),

        # Flatten converts the 2D matrices into a 1D array for the final decision
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        # Softmax outputs 10 probabilities (one for each digit 0-9) that sum to 100%
        layers.Dense(10, activation='softmax')
    ])

    # Compile configures the model for training
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    print("4. Training CNN Model (this takes a moment)...")
    # Epochs=5 means the model will look at the entire dataset 5 times to learn
    model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.1)

    print("5. Evaluating CNN Model...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)

    print(f"Model Performance:")
    print(f"Test Accuracy: {test_acc:.4f}")

    print("6. Saving the trained model...")
    # Save the model architecture and trained weights
    os.makedirs(os.path.join(os.path.dirname(__file__), '../models'), exist_ok=True)
    save_path = os.path.join(os.path.dirname(__file__), '../models/mnist_cnn_model.keras')
    model.save(save_path)
    print(f"Model saved to {save_path}")


def interactive_cnn_prediction():
    model_path = os.path.join(os.path.dirname(__file__), '../models/mnist_cnn_model.keras')

    if not os.path.exists(model_path):
        print("\nError: Model not found. Please run the training pipeline first.")
        return

    print("\n--- CNN Digit Recognizer ---")
    print("Loading model and fetching a random test image...")

    # Load the saved model and the dataset
    model = tf.keras.models.load_model(model_path)
    (_, _), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Pick a random image from the 10,000 test images
    idx = random.randint(0, len(X_test) - 1)
    image = X_test[idx]
    actual_label = y_test[idx]

    print("\nTarget Image (Terminal ASCII Representation):")
    print("-" * 56)
    # Render the 28x28 pixel image in the terminal using ASCII characters
    for row in image:
        line = "".join(["██" if pixel > 128 else "▒▒" if pixel > 64 else "  " for pixel in row])
        print(line)
    print("-" * 56)

    # Preprocess the single image exactly like we did during training
    processed_image = image.astype("float32") / 255.0
    # Add batch dimension and color channel: shape becomes (1, 28, 28, 1)
    processed_image = np.expand_dims(processed_image, axis=(0, -1))

    # Run the prediction
    predictions = model.predict(processed_image, verbose=0)
    predicted_label = np.argmax(predictions)
    confidence = np.max(predictions) * 100

    print("=====================================================")
    print(f"  ACTUAL LABEL:   {actual_label}")
    print(f"  MODEL PREDICTS: {predicted_label} (Confidence: {confidence:.2f}%)")
    print("=====================================================\n")

if __name__ == "__main__":
    run_mnist_cnn()