# ViT-Transformers

This repository demonstrates how to implement, train, and evaluate a Vision Transformer for an image classification task on a custom dataset.

## Key Components

### Import Statements
Import necessary libraries such as NumPy, Pandas, TensorFlow, and Matplotlib.

### Configuration Class (Config)
Stores various configuration parameters used throughout the model, including learning rate, batch size, number of epochs, etc.

### Data Loading and Preprocessing
- Load the CIFAR-10 dataset.
- Extract data related to only three selected classes for training and testing.

### Data Augmentation Layer
Define a sequence of data preprocessing and augmentation operations to increase the robustness of the model.

### Multi-Layer Perceptron (MLP) Function
Create a simple MLP with a specified number of hidden layers.

### Patches and PatchEncoder Classes
- Divide the input image into small patches.
- Encode these patches into a form that can be processed by the transformer layers.

### Vision Transformer (ViT) Model
The `create_vision_transformer` function defines the main model, constructing a transformer-based architecture for image classification tasks.

### Model Compilation and Training
- Compile the model with the AdamW optimizer and Sparse Categorical Crossentropy as the loss function.
- Train the model on the CIFAR-10 data.

### Model Evaluation
- Evaluate the model's performance based on accuracy and loss during training and validation.
- Visualize these metrics over the epochs with plots.
- Display the model's accuracy and loss on the test data in a bar plot.

## Installation

To install the necessary dependencies, ensure you have the following:

```sh
conda create --name vit-transformers python=3.8.11
conda activate vit-transformers
pip install tensorflow-gpu==2.4.1
pip install keras==2.4.3
pip install keras-applications==1.0.8
