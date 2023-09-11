def configure_gpu():
    import tensorflow as tf
    if tf.config.list_physical_devices('GPU'):
        print("GPU is available.")
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("GPU is not available.")
configure_gpu()

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix, classification_report

## Constants & Hyperparameters
num_classes = 3
input_shape = (128, 128, 3)
batch_size = 256
train_data_dir = "Path to Train_Dataset"
validation_data_dir = "Path to Val_dataset"
test_data_dir = "Path to Test_dataset"

#Parameters
learning_rate = 0.001
weight_decay = 0.0001
image_size = 128
patch_size = 8
num_patches = (image_size // patch_size) ** 2
projection_dim = 32
num_heads = 4
transformer_units = [projection_dim * 2, projection_dim]
transformer_layers = 8
mlp_head_units = [2048, 1024]
dropout_rate = 0.5 
num_epochs = 500

# Initialize the MirroredStrategy to use both gpu
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu, kernel_regularizer=keras.regularizers.l2(weight_decay))(x)
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
        return tf.reshape(patches, [batch_size, -1, patch_dims])

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patches):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patches) + self.position_embedding(positions)
        return encoded

def create_vit_classifier():
    inputs = layers.Input(shape=input_shape)
    augmented = data_augmentation(inputs)
    patches = Patches(patch_size)(augmented)
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    for _ in range(transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        x2 = layers.Add()([attention_output, encoded_patches])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        encoded_patches = layers.Add()([x3, x2])

    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    logits = layers.Dense(num_classes)(features)
    return keras.Model(inputs=inputs, outputs=logits)

def run_experiment(model):
    decayed_learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=1000,
        decay_rate=0.9,
        staircase=True
    )
    
    optimizer = keras.optimizers.Adam(learning_rate=decayed_learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.CategoricalAccuracy(name="accuracy")
        ],
    )

    checkpoint_filepath = "/tmp/checkpoint"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )

    history = model.fit(
        train_generator,
        epochs=num_epochs,
        validation_data=validation_generator, 
        callbacks=[checkpoint_callback],
    )

    model.load_weights(checkpoint_filepath)
    _, accuracy = model.evaluate(test_generator) 
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    final_train_accuracy = history.history['accuracy'][-1]
    final_val_accuracy = history.history['val_accuracy'][-1]
    print(f"Final training accuracy: {round(final_train_accuracy * 100, 2)}%")
    print(f"Final validation accuracy: {round(final_val_accuracy * 100, 2)}%")

    return history

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
  
    plt.figure(figsize=(4,4))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def evaluate_model(model, test_generator):
    # getting the true labels and predictions
    y_true = test_generator.classes
    y_pred = model.predict(test_generator, verbose=1)
    y_pred_classes = np.argmax(y_pred, axis=1)

    cm = confusion_matrix(y_true, y_pred_classes)
    plot_confusion_matrix(cm, classes=test_generator.class_indices.keys())
    
    # to print classification report
    print(classification_report(y_true, y_pred_classes, target_names=test_generator.class_indices.keys()))

if __name__ == "__main__":
    
    configure_gpu()
    train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    validation_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )

    validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    # Visualization code 
    x_train_batch, _ = next(train_generator)
    plt.figure(figsize=(4, 4))
    image = x_train_batch[np.random.choice(range(x_train_batch.shape[0]))]
    plt.imshow(image)
    plt.axis("off")
    plt.show() 

    resized_image = tf.image.resize(
        tf.convert_to_tensor([image]), size=(image_size, image_size)
    )
    patches = Patches(patch_size)(resized_image)
    
    n = int(np.sqrt(patches.shape[1]))
    plt.figure(figsize=(4, 4))
    for i, patch in enumerate(patches[0]):
        ax = plt.subplot(n, n, i + 1)
        patch_img = tf.reshape(patch, (patch_size, patch_size, 3))
        plt.imshow(patch_img.numpy())
        plt.axis("off")
    plt.show()

    print(f"Train samples: {train_generator.samples}, Test samples: {test_generator.samples}")

    with strategy.scope():
        # Model-related code
        data_augmentation = keras.Sequential(
            [
                layers.Lambda(lambda x: (x - [0.5, 0.5, 0.5]) / [0.5, 0.5, 0.5]),
                layers.experimental.preprocessing.Resizing(image_size, image_size),
                layers.experimental.preprocessing.RandomFlip("horizontal"),
                layers.experimental.preprocessing.RandomRotation(factor=0.02),
                layers.experimental.preprocessing.RandomZoom(height_factor=0.2, width_factor=0.2)
            ],
            name="data_augmentation"
        )

        vit_classifier = create_vit_classifier()
        history = run_experiment(vit_classifier)

    evaluate_model(vit_classifier, test_generator) 

