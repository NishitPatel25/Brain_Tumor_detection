# Brain Tumor Detection

This project focuses on detecting brain tumors from MRI images using Convolutional Neural Networks (CNN) with TensorFlow Keras. The model is trained on a dataset of brain tumor images sourced from Kaggle, and it classifies whether an MRI scan indicates the presence of a tumor or not.

## Features

- **Brain Tumor Detection:** Classifies MRI images as either tumor or non-tumor.
- **CNN Model:** Uses a Convolutional Neural Network for image feature extraction and classification.
- **TensorFlow + Keras:** Model implementation and training on MRI image dataset.
- **Image Visualization:** Displays sample MRI images and the model’s prediction accuracy.

## Technology Stack

- **Kaggle MRI Dataset:** A dataset of MRI brain images, including labeled data for tumor detection.
- **CNN Model:** A deep learning model designed for image classification tasks.
- **TensorFlow + Keras:** Backend framework for building and training the CNN model.
- **NumPy and Pandas:** For data handling and preprocessing.
- **Matplotlib:** For visualizing MRI images and model performance metrics.

## Prerequisites

Before running the project, ensure you have the following installed:

- Python 3.x
- TensorFlow and Keras (`pip install tensorflow`)
- NumPy (`pip install numpy`)
- Pandas (`pip install pandas`)
- Matplotlib (`pip install matplotlib`)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/NishitPatel25/brain-tumor-detection.git
    ```
2. Navigate to the project directory:
    ```bash
    cd brain-tumor-detection
    ```
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

1. Download the brain tumor image dataset from Kaggle.
2. Preprocess the image data (resizing, normalizing, etc.).
3. Train the CNN model using the dataset by running the `train_model.py` file:
    ```bash
    python train_model.py
    ```
4. Test the trained model on new MRI images to predict if a tumor is present:
    ```bash
    python test_model.py
    ```

## Dataset

The dataset used for this project consists of MRI brain images labeled as either "tumor" or "non-tumor." The dataset can be found on [Kaggle](https://www.kaggle.com/). Download and extract the dataset into the appropriate directory before running the model.

## Usage

- Preprocess the images by resizing and normalizing them for input into the CNN model.
- Train the CNN model to classify MRI images as containing a tumor or not.
- Visualize the predictions, accuracy, and loss using Matplotlib.

## Model Overview

The CNN architecture includes:
- **Convolutional Layers:** To extract spatial features from MRI images.
- **Max Pooling Layers:** To reduce the spatial dimensions of the image and computational cost.
- **Dense Layers:** For final classification into tumor or non-tumor categories.

## Visualization

Matplotlib is used to visualize:
- Sample MRI images with predicted labels.
- Model training accuracy and loss.
- Confusion matrix and other evaluation metrics.

## Future Work

- Improve model accuracy by experimenting with different CNN architectures.
- Add data augmentation to increase the robustness of the model.
- Implement a web interface to upload MRI scans for real-time tumor detection.
