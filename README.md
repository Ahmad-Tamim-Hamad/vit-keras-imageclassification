# Vision Transformer (ViT) Classifier for Image Classification

Welcome to the **Vision Transformer (ViT) Classifier** project. This repository contains the implementation of a Vision Transformer (ViT) model for image classification. The project is designed to work with any dataset structured in image directories and is equipped with training, validation, and testing functionalities. It also provides tools for evaluating model performance through confusion matrices and classification reports.

## Table of Contents
- [About the Project](#about-the-project)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Evaluating the Model](#evaluating-the-model)
  - [Customizing the Model](#customizing-the-model)
- [File Structure](#file-structure)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## About the Project

This project implements a **Vision Transformer (ViT)** architecture for image classification tasks. It uses TensorFlow and Keras for model construction, training, and evaluation. The project is designed to be scalable across multiple GPUs using TensorFlow's `MirroredStrategy`.

The main objectives are:
- To classify images using a transformer-based architecture.
- To experiment with a variety of hyperparameters and augmentation techniques.
- To allow easy usage and customization for any user interested in image classification with transformers.

## Features
- **Vision Transformer Architecture**: Utilizes patches and transformer layers to process images.
- **Data Augmentation**: Includes horizontal flipping, rotation, and zoom for enhancing training data.
- **Multi-GPU Support**: Leverages TensorFlow's `MirroredStrategy` for distributed training across multiple GPUs.
- **Evaluation Tools**: Generates confusion matrices and classification reports.
- **Configurable Hyperparameters**: Users can adjust image size, patch size, number of transformer layers, etc.

## Getting Started

### Prerequisites
Before you begin, ensure that you have the following installed:
- Python 3.8+
- TensorFlow 2.5 or newer
- GPU support (CUDA-enabled GPU recommended for faster training)
- Required Python libraries (listed in `requirements.txt`)

### Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/<your-username>/ViT-Image-Classification.git
    cd ViT-Image-Classification
    ```

2. **Create a virtual environment** (optional but recommended):
    ```bash
    python -m venv vit-env
    source vit-env/bin/activate  # On Windows: vit-env\Scripts\activate
    ```

3. **Install required packages**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Set up GPU (optional)**:
   To use GPU for training, ensure your system has CUDA and cuDNN installed.

## Usage

### Training the Model

1. **Prepare your dataset**:
   Ensure your data is structured as follows:
   ```bash
   Dataset/
   ├── train/
   │   ├── class1/
   │   ├── class2/
   │   └── class3/
   ├── val/
   │   ├── class1/
   │   ├── class2/
   │   └── class3/
   └── test/
       ├── class1/
       ├── class2/
       └── class3/
