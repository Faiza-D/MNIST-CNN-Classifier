# MNIST-CNN-Classifier
A simple convolutional neural network (CNN) model for classifying handwritten digits in the MNIST dataset.

## Dataset Description:

The MNIST dataset is a collection of handwritten digits from 0 to 9. It consists of 60,000 training images and 10,000 testing images, each of size 28x28 pixels. The images are grayscale, meaning they have a single channel.

## Code Description:

The code provided is a TensorFlow implementation of a convolutional neural network (CNN) model for classifying handwritten digits in the MNIST dataset. Here's a breakdown of the key steps:

1. **Load the Dataset:** The code loads the MNIST dataset using `tf.keras.datasets.mnist.load_data()`, separating it into training and testing sets.
2. **Preprocess the Data:** The images are normalized to the range [0, 1] and reshaped to add a single channel dimension.
3. **Build the CNN Model:** A simple CNN architecture is defined using `tf.keras.models.Sequential`. It consists of convolutional layers with ReLU activation, max pooling layers, and fully connected layers.
4. **Compile the Model:** The model is compiled with the Adam optimizer, sparse categorical crossentropy loss, and accuracy metric.
5. **Train the Model:** The model is trained on the training data for 5 epochs, with validation on the testing data.
6. **Evaluate the Model:** The model's performance is evaluated on the testing data, and the test accuracy is printed.
7. **Visualize Training and Validation Accuracy:** The training and validation accuracy over epochs are plotted to assess the model's learning progress.

The code effectively demonstrates how to build and train a CNN model for image classification using TensorFlow, providing a solid foundation for further exploration and experimentation.

