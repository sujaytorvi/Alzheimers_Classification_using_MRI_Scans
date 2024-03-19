import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint, EarlyStopping
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Function to set up TPU if available
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

# Function to load image datasets from directory
def load_image_dataset(directory, subset):
    return tf.keras.preprocessing.image_dataset_from_directory(
        directory,
        validation_split=0.2,
        subset=subset,
        seed=2024,
        image_size=[176, 208],
        batch_size=16 * strategy.num_replicas_in_sync,
    )

# Function to one-hot encode labels
def one_hot_label(image, label):
    label = tf.one_hot(label, 4)
    return image, label

# Function to get number of images for each class
def get_num_images(class_names):
    NUM_IMAGES = []
    for label in class_names:
        dir_name = f"/Users/sujaymukundtorvi/Desktop/ALY 6980 Capstone/alzheimers_dataset/Dataset/{label[:-2]}ed"
        NUM_IMAGES.append(len([name for name in os.listdir(dir_name)]))
    return NUM_IMAGES

# CNN model definition functions
def conv_block(filters):
    block = tf.keras.Sequential([
        layers.SeparableConv2D(filters, 3, activation='relu', padding='same'),
        layers.SeparableConv2D(filters, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPool2D()
    ])
    return block

def dense_block(units, dropout_rate):
    block = tf.keras.Sequential([
        layers.Dense(units, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate)
    ])
    return block

def build_model():
    model = tf.keras.Sequential([
        layers.experimental.preprocessing.Resizing(150, 150),
        layers.Conv2D(16, 3, activation='relu', padding='same'),
        layers.Conv2D(16, 3, activation='relu', padding='same'),
        layers.MaxPool2D(),
        conv_block(32),
        conv_block(64),
        conv_block(128),
        layers.Dropout(0.2),
        conv_block(256),
        layers.Dropout(0.2),
        layers.Flatten(),
        dense_block(512, 0.7),
        dense_block(128, 0.5),
        dense_block(64, 0.3),
        layers.Dense(4, activation='softmax')
    ])
    return model

# Function for exponential decay of learning rate
def exponential_decay(lr0, s):
    def exponential_decay_fn(epoch):
        return lr0 * 0.1 ** (epoch / s)
    return exponential_decay_fn

# Function to preprocess an image
def preprocess_image(image_path, target_size=(176, 208)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to plot images with predictions
def plot_images_predictions(dataset, class_names, model):
    plt.figure(figsize=(10, 10))

    for images, labels in dataset.unbatch().batch(9).take(1):
        predictions = model.predict(images)
        predicted_class_indices = np.argmax(predictions, axis=1)
        confidences = np.max(predictions, axis=1)

        for i in range(9):
            plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.axis("off")

            actual_idx = np.argmax(labels[i].numpy()) if labels[i].numpy().size > 1 else labels[i].numpy()
            actual_label = class_names[actual_idx]
            predicted_label = class_names[predicted_class_indices[i]]
            confidence = confidences[i]

            plt.title(f"Actual: {actual_label}\nPredicted: {predicted_label}\nConfidence: {confidence:.2f}")

    plt.tight_layout()
    plt.show()


# Constants and setup
tf.get_logger().setLevel('ERROR')
strategy = setup_tpu()
AUTOTUNE = tf.data.experimental.AUTOTUNE
class_names = ['Mild_Dementia', 'Moderate_Dementia', 'Non_Dementia', 'Very_Mild_Dementia']
NUM_IMAGES = get_num_images(class_names)

# Load datasets
train_ds = load_image_dataset("/Users/sujaymukundtorvi/Desktop/ALY 6980 Capstone/alzheimers_dataset/Dataset", "training")
val_ds = load_image_dataset("/Users/sujaymukundtorvi/Desktop/ALY 6980 Capstone/alzheimers_dataset/Dataset", "validation")

# Apply preprocessing and one-hot encoding
train_ds = train_ds.map(one_hot_label, num_parallel_calls=AUTOTUNE).cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.map(one_hot_label, num_parallel_calls=AUTOTUNE).cache().prefetch(buffer_size=AUTOTUNE)

# Build and compile the model using TPU strategy
with strategy.scope():
    model = build_model()
    METRICS = [tf.keras.metrics.AUC(name='auc'), tf.keras.metrics.CategoricalAccuracy(name='acc'), tf.metrics.F1Score()]
    model.compile(
        optimizer='adam',
        loss=tf.losses.CategoricalCrossentropy(),
        metrics=METRICS
    )

# Training and callbacks
EPOCHS = 45
BATCH_SIZE = 16
lr_scheduler = LearningRateScheduler(exponential_decay(0.01, 20))
checkpoint_cb = ModelCheckpoint("alzheimer_model.h5", save_best_only=True)
early_stopping_cb = EarlyStopping(patience=15, restore_best_weights=True)
history = model.fit(train_ds, validation_data=val_ds, callbacks=[checkpoint_cb, early_stopping_cb, lr_scheduler], epochs=EPOCHS)

# Evaluate the model on the test dataset
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "/Users/sujaymukundtorvi/Desktop/ALY 6980 Capstone/Alzheimer_s Dataset/Test/",
    image_size=(176, 208),
    batch_size=BATCH_SIZE,
)
test_ds = test_ds.map(one_hot_label, num_parallel_calls=AUTOTUNE).cache().prefetch(buffer_size=AUTOTUNE)
_ = model.evaluate(test_ds)

# Load the trained model
model_path = '/Users/sujaymukundtorvi/Desktop/ALY 6980 Capstone/Best Models/alzheimer_cnn_model_best.h5'
model = tf.keras.models.load_model(model_path)

# Plot images with predictions
plot_images_predictions(val_ds,class_names, model)

# Generate and display confusion matrix
y_true = []
y_pred = []

for img_batch, label_batch in val_ds:
    preds = model.predict(img_batch, verbose=False)
    pred_labels = np.argmax(preds, axis=1)
    y_pred.extend(pred_labels)

    true_labels = np.argmax(label_batch.numpy(), axis=1)
    y_true.extend(true_labels)

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
