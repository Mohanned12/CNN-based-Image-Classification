### CNN-based-Image-Classification




1. INTRODUCTION

      This repository contains the implementation of a Convolutional Neural Network (CNN) for image classification on CIFAR-10 using PyTorch. This project serves as an introduction to deep              learning and computer vision.

2. DATASET - CIFAR-10

      CIFAR-10 contains 60,000 images in 10 classes (e.g., airplane, car, bird, cat, etc.).

      Each image is of size 32x32 pixels with 3 color channels (RGB).

      The dataset is divided into 50,000 training images and 10,000 test images.

      DATA PREPROCESSING:

      Images are converted to tensors.

      Normalization is applied to scale pixel values between -1 and 1.

      Batch size is set to 128 for training and testing.

3. CNN MODEL ARCHITECTURE

      The model consists of three convolutional layers:

            Conv1: 3 input channels → 32 output channels (kernel size 3x3)

            Conv2: 32 input channels → 64 output channels (kernel size 3x3)

            Conv3: 64 input channels → 128 output channels (kernel size 3x3)

      Each convolutional layer is followed by ReLU activation and Max Pooling (2x2).

      Fully connected layers:

            FC1: 128 * 4 * 4 → 256

            FC2: 256 → 64

            FC3: 64 → 10 (corresponding to the 10 CIFAR-10 classes)

4. TRAINING PROCESS

      Loss Function: CrossEntropyLoss (suitable for multi-class classification)

      Optimizer: Adam optimizer with a learning rate of 0.001

      Training Strategy:

            Forward pass: Compute predictions.

            Compute loss.

            Backward pass: Compute gradients.

            Update weights using optimizer.

      Training runs for 10 epochs.

      Training loss decreases progressively, indicating learning progress.

5. MODEL EVALUATION

      The model is tested on 10,000 images.

      Accuracy Calculation:

            Predictions are obtained using `` on model output.

            Correct predictions are counted, and accuracy is computed.

      Results:

            Training loss decreases across epochs.
      
            Final test accuracy achieved: 74.56%.

6. CONCLUSION

      Successfully implemented a CNN for CIFAR-10 image classification.
      Model achieves reasonable accuracy for a simple architecture.
