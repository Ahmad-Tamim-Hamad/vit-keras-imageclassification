# ViT-Transformers.
This model demonstrates how to implement, train, and evaluate a Vision Transformer for an image classification task on a subset of the CIFAR-10 dataset.

## Key parts of the code include
1. Import Statements: It imports necessary libraries such as numpy, pandas, TensorFlow, and matplotlib.
2. Configuration Class (Config): This class is used to store various configuration parameters used throughout the model like learning rate, batch size, number of epochs, etc.
3. Data Loading and Preprocessing: The CIFAR-10 dataset is loaded, and data related to only three selected classes is extracted for training and testing.
4. Data Augmentation Layer: A sequence of data preprocessing and augmentation operations are defined to increase the robustness of the model.
5. Multi-Layer Perceptron (MLP) Function: This function creates a simple MLP with a specified number of hidden layers.
6. Patches and PatchEncoder Classes: These classes are used to divide the input image into small patches, and then encode these patches into a form that can be processed by the transformer layers.
7. Vision Transformer (ViT) Model: The function create_vision_transformer defines the main model. It constructs a transformer-based model architecture for image classification tasks.
8. Model Compilation and Training: The model is compiled with the AdamW optimizer and Sparse Categorical Crossentropy as the loss function, and then trained on the CIFAR-10 data.
9. Model Evaluation: The model's performance is evaluated based on accuracy and loss during training and validation. Plots are made to visualize these metrics over the epochs. At the end, the model's accuracy and loss on the test data are shown in a bar plot.

## Install
!pip install tqdm
!pip install transformers datasets
!pip install torchinfo
!pip install --upgrade tensorflow-addons
