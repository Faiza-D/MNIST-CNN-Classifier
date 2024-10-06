import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# Normalize the images to the range [0, 1]
train_images, test_images = train_images / 255.0, test_images / 255.0

# Reshape images to add a single channel (grayscale)
train_images = train_images.reshape(-1, 28, 28, 1)
test_images = test_images.reshape(-1, 28, 28, 1)

print("Training Data Shape:", train_images.shape)

# Build a simple CNN model for MNIST
model = models.Sequential([
  layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
  layers.MaxPooling2D((2, 2)),
  layers.Conv2D(64, (3, 3), activation='relu'),
  layers.MaxPooling2D((2, 2)),
  layers.Flatten(),  # Flatten the 3D output to 1D
  layers.Dense(64, activation='relu'),
  layers.Dense(10)  # Output layer for 10 classes (0-9 digits)
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model for 5 epochs
history = model.fit(train_images, train_labels, epochs=5,
                    validation_data=(test_images, test_labels))

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"Test accuracy: {test_acc}")

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

# Make predictions on a few test images
predicted_classes = model.predict_classes(test_images[:10])  # Predict for first 10 images

# Make predictions on a few test images
predicted_probs = model.predict(test_images[:10])  # Predict for first 10 images

# Get the predicted classes by taking the index of the maximum probability
predicted_classes = predicted_probs.argmax(axis=-1)

# Print the predicted labels and true labels for comparison
for i, (image, label, prediction) in enumerate(zip(test_images[:10], test_labels[:10], predicted_classes)):
    plt.imshow(image[:, :, 0], cmap='gray')  # Assuming grayscale images
    plt.title(f"True Label: {label}, Predicted: {prediction}")
    plt.show()