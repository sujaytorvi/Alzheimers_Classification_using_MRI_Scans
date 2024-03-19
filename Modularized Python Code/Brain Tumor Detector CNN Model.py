import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint, EarlyStopping
import seaborn as sns
from keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.optimizers import RMSprop
from sklearn.metrics import classification_report, confusion_matrix

# Function to set up TPU (Tensor Processing Unit) if available
def setup_tpu():
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print('Device:', tpu.master())
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
    except:
        strategy = tf.distribute.get_strategy()
    return strategy

# Constants
AUTOTUNE = tf.data.experimental.AUTOTUNE
NUM_CLASSES = 4  # Number of classes
IMAGE_SIZE = [176, 208]
BATCH_SIZE = 16 * setup_tpu().num_replicas_in_sync
IMAGE_DIM = 150
EPOCHS = 100

# Function to load image datasets from directory
def load_image_dataset(directory, subset):
    return tf.keras.preprocessing.image_dataset_from_directory(
        directory,
        validation_split=0.2,
        subset=subset,
        seed=2024,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
    )

# Load training and validation datasets
train_ds = load_image_dataset("/Users/sujaymukundtorvi/Desktop/ALY 6980 Capstone/archive/brain_tumor_dataset", "training")
val_ds = load_image_dataset("/Users/sujaymukundtorvi/Desktop/ALY 6980 Capstone/archive/brain_tumor_dataset", "validation")

# Define class names
class_names = ['no', 'yes']
train_ds.class_names = class_names
val_ds.class_names = class_names

# Display sample images
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(train_ds.class_names[labels[i]])
        plt.axis("off")

# Function to one-hot encode labels
def one_hot_label(image, label):
    label = tf.one_hot(label, NUM_CLASSES)
    return image, label

# Apply one-hot encoding to datasets
train_ds = train_ds.map(one_hot_label, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(one_hot_label, num_parallel_calls=AUTOTUNE)

# Cache and prefetch datasets for better performance
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Function to resize images
def resize_images(image, label):
    resized_image = tf.image.resize(image, IMAGE_SIZE)
    return resized_image, label

# Apply resizing to validation dataset
val_ds_resized = val_ds.map(resize_images, num_parallel_calls=tf.data.experimental.AUTOTUNE)

def define_VGG_NET():
    # load base model
    vgg16_weight_path = 'imagenet'
    base_model = VGG16(
        weights=vgg16_weight_path,
        include_top=False, 
        input_shape=IMAGE_SIZE + (3,)
    )
    return base_model

# Function to build the model for binary classification
def build_binary_model():
    model = tf.keras.Sequential()
    model.add(define_VGG_NET())
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(NUM_CLASSES, activation='sigmoid'))

    model.layers[0].trainable = False
    METRICS = [tf.keras.metrics.AUC(name='auc'), tf.keras.metrics.CategoricalAccuracy(name='acc'), tf.metrics.F1Score()]

    model.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.Adam(lr=0.0001),
    metrics=METRICS
    )

    model.summary()
    return model

# Build and compile the model using TPU strategy
with setup_tpu().scope():
    model = build_binary_model()
    METRICS = [tf.keras.metrics.AUC(name='auc'), tf.keras.metrics.CategoricalAccuracy(name='acc'), tf.metrics.F1Score()]
    model.compile(
        optimizer='adam',
        loss=tf.losses.CategoricalCrossentropy(),
        metrics=METRICS
    )

# Model checkpoint callback
checkpoint_cb = ModelCheckpoint("brain_tumor_cnn_model.h5", save_best_only=True)

# Early stopping callback
early_stopping_cb = EarlyStopping(patience=10, restore_best_weights=True)

history = model.fit_generator(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=[checkpoint_cb, early_stopping_cb]
)

# Assign variables
EPOCHS = 50 #Early stopping epoch value
auc = history.history['auc']
val_auc = history.history['val_auc']
loss = history.history['loss']
val_loss = history.history['val_loss']

# Plot training and validation accuracies
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(range(EPOCHS), auc, label='Training Accuracy')
plt.plot(range(EPOCHS), val_auc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

# Plot training and validation losses
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(range(EPOCHS), loss, label='Training Loss')
plt.plot(range(EPOCHS), val_loss, label='Validation Loss')
plt.legend(loc='lower right')
plt.title('Training and Validation Loss')
plt.show()

# Define a function to preprocess an image
def preprocess_image(image_path, target_size=(176, 208)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Collect true and predicted labels for confusion matrix and classification report
y_true = []
y_pred = []

for img_batch, label_batch in val_ds_resized:
    # Predict
    preds = model.predict(img_batch, verbose=False)
    pred_labels = np.argmax(preds, axis=1)
    y_pred.extend(pred_labels)

    # Convert one-hot encoded labels to indices for true labels
    true_labels = np.argmax(label_batch.numpy(), axis=1)
    y_true.extend(true_labels)

# Plot images with predictions
def plot_images_predictions(dataset, class_names, model):
    plt.figure(figsize=(10, 10))

    # Unbatching dataset and taking 9 samples
    for images, labels in dataset.u.imshow(images[i].numpy().astype("uint8"))
            plt.axis("off")
 # Adjust for one-hot encoded labels or tensor labels
            actual_idx = np.argmax(labels[i].numpy()) if labels[i].numpy().size > 1 else labels[i].numpy()
            actual_label = class_names[actual_idx]
            predicted_label = class_names[predicted_class_indices[i]]
            confidence = confidences[i]

            plt.title(f"Actual: {actual_label}\nPredicted: {predicted_label}\nConfidence: {confidence:.2f}")

    plt.tight_layout()
    plt.show()

# Generate and display confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

# Generate and display the classification report
report = classification_report(y_true, y_pred, target_names=class_names)
print(report)
