# ViT-Transformers
This model demonstrates how to implement, train, and evaluate a Vision Transformer for image classification task on a custom dataset

## Key parts of the code include
* Import Statements: It imports necessary libraries such as numpy, pandas, TensorFlow, and matplotlib.
* Configuration Class (Config): This class is used to store various configuration parameters used throughout the model like learning rate, batch size, number of epochs, etc.
* Data Loading and Preprocessing: The CIFAR-10 dataset is loaded, and data related to only three selected classes is extracted for training and testing.
* Data Augmentation Layer: A sequence of data preprocessing and augmentation operations are defined to increase the robustness of the model.
* Multi-Layer Perceptron (MLP) Function: This function creates a simple MLP with a specified number of hidden layers.
* Patches and PatchEncoder Classes: These classes are used to divide the input image into small patches, and then encode these patches into a form that can be processed by the transformer layers.
* Vision Transformer (ViT) Model: The function create_vision_transformer defines the main model. It constructs a transformer-based model architecture for image classification tasks.
* Model Compilation and Training: The model is compiled with the AdamW optimizer and Sparse Categorical Crossentropy as the loss function, and then trained on the CIFAR-10 data.
* Model Evaluation: The model's performance is evaluated based on accuracy and loss during training and validation. Plots are made to visualize these metrics over the epochs. At the end, the model's accuracy and loss on the test data are shown in a bar plot.

## Install
* python=3.8.11
* tensorflow-gpu=2.4.1
* keras==2.4.3
* keras-applications==1.0.8

## Note:
* Given the sizable nature of my custom dataset and the fact that each image is approximately 128 pixels, I utilized TensorFlow's distributed training.

