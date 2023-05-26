import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import tensorflow_addons as tfa

class Config:
    input_size = 32
    input_shape = [input_size, input_size, 3]
    learning_rate = 0.001
    weight_decay = 0.0001
    batch_size = 256
    num_classes = 3
    num_epochs = 100
    image_size = 72
    patch_size = 6
    num_patches = (image_size // patch_size) ** 2
    projection_dim = 64
    num_heads = 4
    transformer_units = [
        projection_dim * 2,
        projection_dim
    ]
    transformer_layers = 8
    mlp_head_units = [2048, 1024]

(train_data, train_labels), (test_data, test_labels) = keras.datasets.cifar10.load_data()

# Select only three classes, for example classes 0, 1, and 2
selected_classes = [0, 1, 2]
train_indices = np.isin(train_labels, selected_classes).flatten()
test_indices = np.isin(test_labels, selected_classes).flatten()

train_data, train_labels = train_data[train_indices], train_labels[train_indices]
test_data, test_labels = test_data[test_indices], test_labels[test_indices]

print("Training data shape:", train_data.shape, train_labels.shape)
print("Test data shape:", test_data.shape, test_labels.shape)

augmentation_layer = tf.keras.Sequential([
    keras.layers.Input(Config.input_shape),
    keras.layers.experimental.preprocessing.Normalization(),
    keras.layers.experimental.preprocessing.Resizing(Config.image_size, Config.image_size),
    keras.layers.experimental.preprocessing.RandomRotation(factor=0.02),
    keras.layers.experimental.preprocessing.RandomZoom(height_factor=0.2, width_factor=0.2),
])

augmentation_layer.layers[0].adapt(train_data)

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

def create_vision_transformer():
    inputs = layers.Input(shape=Config.input_shape)
    augmented = augmentation_layer(inputs)

    patches = Patches(Config.patch_size)(augmented)
    encoder_patches = PatchEncoder(Config.num_patches, Config.projection_dim)(patches)

    for _ in range(Config.transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoder_patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=Config.num_heads,
            key_dim=Config.projection_dim,
            dropout=0.1
        )(x1, x1)
        x2 = layers.Add()([attention_output, encoder_patches])

        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp(x3, hidden_units=Config.transformer_units, dropout_rate=0.1)
        encoder_patches = layers.Add()([x3, x2])

    representation = layers.LayerNormalization(epsilon=1e-6)(encoder_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)

    features = mlp(representation, hidden_units=Config.mlp_head_units, dropout_rate=0.5)
    outputs = layers.Dense(Config.num_classes)(features)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

keras.backend.clear_session()
vit_classifier = create_vision_transformer()
vit_classifier.summary()

optimizer = tfa.optimizers.AdamW(
    learning_rate=Config.learning_rate,
    weight_decay=Config.weight_decay
)

vit_classifier.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=optimizer,
    metrics=["accuracy"]
)

checkpoint_path = "model.h5"
checkpoint = keras.callbacks.ModelCheckpoint(
    checkpoint_path,
    monitor="val_accuracy",
    save_best_only=True,
    save_weights_only=True
)

history = vit_classifier.fit(
    train_data,
    train_labels,
    validation_data=(test_data, test_labels),
    batch_size=Config.batch_size,
    epochs=Config.num_epochs,
    callbacks=[checkpoint],
)

# Plot the training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Add space between the two plots
plt.figure()

plt.figure()

# Plot the training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

import matplotlib.pyplot as plt

# Set a style for the plots
plt.style.use('seaborn-darkgrid')

# Get the test accuracy and test loss
test_accuracy = history.history['val_accuracy'][-1]
test_loss = history.history['val_loss'][-1]

# Create a bar plot with two bars
fig, ax = plt.subplots()
ax.bar(['Test Accuracy'], [test_accuracy], color='dodgerblue', label='Test Accuracy')
ax.bar(['Test Loss'], [test_loss], color='darkorange', label='Test Loss')

# Add text labels to the bars
ax.text(0, test_accuracy, f"{test_accuracy:.2f}", ha='center', va='bottom', fontweight='bold')
ax.text(1, test_loss, f"{test_loss:.4f}", ha='center', va='bottom', fontweight='bold')

# Set the y-axis limits
ax.set_ylim(0, max(test_accuracy, test_loss) * 1.1)

# Add labels and title
plt.xlabel('Metrics')
plt.ylabel('Value')
plt.title('Test Accuracy and Test Loss')
plt.legend()

# Show the plot
plt.show()



