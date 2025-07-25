# CIFAR-10 Image Classification with CNN

This project demonstrates how to build and train a Convolutional Neural Network (CNN) using TensorFlow and Keras for image classification on the CIFAR-10 dataset. The script also includes an optional section for transfer learning using a pre-trained VGG16 model.

## Prerequisites and Dependencies

To run this script, you need Python installed along with the following libraries:

*   **TensorFlow:** `pip install tensorflow`
*   **NumPy:** `pip install numpy`
*   **Scikit-learn:** `pip install scikit-learn`

Make sure you have these installed in your Python environment.

## Script Functionality

The script performs the following steps:

1.  **Load and Preprocess the CIFAR-10 Dataset:**
    *   Loads the CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 classes.
    *   Normalizes the pixel values of the images to be between 0 and 1.
    *   Converts the integer labels to one-hot encoded categorical vectors.

2.  **Build the CNN Model:**
    *   Defines a sequential CNN model.
    *   The architecture includes:
        *   Two convolutional blocks, each with `Conv2D`, `BatchNormalization`, `MaxPooling2D`, and `Dropout` layers.
        *   A flattening layer.
        *   A dense layer with 512 units and ReLU activation.
        *   A batch normalization layer.
        *   A dropout layer.
        *   An output dense layer with `num_classes` (10) units and softmax activation for multi-class classification.
    *   Compiles the model using the Adam optimizer and categorical cross-entropy loss.

3.  **Configure Early Stopping:**
    *   Sets up an `EarlyStopping` callback to monitor the validation loss.
    *   Training will stop if the validation loss does not improve for a specified number of epochs (`patience=10`), and the best model weights will be restored.

4.  **Data Augmentation:**
    *   Uses `ImageDataGenerator` to perform real-time data augmentation on the training images.
    *   Augmentation techniques include random rotations, width and height shifts, and horizontal flips. This helps to prevent overfitting and improve the model's generalization.

5.  **Train the Model:**
    *   Trains the CNN model using the augmented training data.
    *   Specifies the batch size and the maximum number of epochs.
    *   Uses the test data for validation during training.

6.  **Evaluate the Model:**
    *   Evaluates the trained model on the test dataset to determine its accuracy.
    *   Generates and prints a classification report showing precision, recall, and F1-score for each class.

7.  **Optional Bonus: Transfer Learning using VGG16:**
    *   Includes an optional section to demonstrate transfer learning.
    *   Loads the pre-trained VGG16 model (trained on ImageNet) without its top classification layer.
    *   Freezes the weights of the base VGG16 model.
    *   Adds new custom top layers: a `Flatten` layer, a `Dense` layer (256 units, ReLU), a `Dropout` layer, and an output `Dense` layer (softmax).
    *   Compiles and trains this new model using the CIFAR-10 data (with data augmentation).
    *   Evaluates the performance of the transfer learning model.


## About the CIFAR-10 Dataset

The CIFAR-10 dataset is a widely used benchmark dataset in computer vision and machine learning. It consists of 60,000 32x32 pixel color images, categorized into 10 mutually exclusive classes, with 6,000 images per class. The classes are:

*   Airplane
*   Automobile
*   Bird
*   Cat
*   Deer
*   Dog
*   Frog
*   Horse
*   Ship
*   Truck

The dataset is split into a training set of 50,000 images and a test set of 10,000 images.

## Potential Improvements and Future Work

*   **Hyperparameter Tuning:** Experiment with different learning rates, batch sizes, number of epochs, and optimizer settings to potentially improve model performance. Techniques like Grid Search or Random Search can be used.
*   **Different Architectures:** Try other CNN architectures (e.g., ResNet, Inception) or customize the existing one further.
*   **Advanced Data Augmentation:** Explore more sophisticated data augmentation techniques.
*   **Learning Rate Scheduling:** Implement learning rate decay or other scheduling techniques.
*   **Regularization Techniques:** Experiment with other regularization methods like L1/L2 regularization in dense layers.
*   **Full VGG16 Fine-tuning:** Instead of just training the top layers of VGG16, unfreeze some of the later convolutional layers of the base model and fine-tune them with a very small learning rate. This might require resizing CIFAR-10 images to a larger dimension if VGG16's original input size is strictly required for better performance.
*   **Save and Load Model:** Add functionality to save the trained model weights and load them later for inference without retraining.
