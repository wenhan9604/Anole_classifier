import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from keras.optimizers import Adam
from utility import *
from sklearn.metrics import classification_report
import numpy as np

"""
Training for the one-vs all ensemble of classifiers. Testing at the end of training.
"""

ensemble_models = None

# Function to build individual models
def build_model():
    base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), 
                                                   include_top=False, 
                                                   weights='imagenet')
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')  # Binary output for one class
    ])
    return model

def balance_class_dataset(dataset, class_index):
    # Filter positives and assign binary label 1
    positives = dataset.filter(lambda image, label, *args: tf.equal(label[class_index], 1))
    positives = positives.map(lambda image, label, *args: (image, tf.constant(1, dtype=tf.int64)))

    # Filter negatives and assign binary label 0
    negatives = dataset.filter(lambda image, label, *args: tf.equal(label[class_index], 0))
    negatives = negatives.map(lambda image, label, *args: (image, tf.constant(0, dtype=tf.int64)))

    # Determine the minimum number of samples
    num_positives = positives.cardinality().numpy()
    num_negatives = negatives.cardinality().numpy()
    min_samples = tf.constant(min(num_positives, num_negatives), dtype=tf.int64)

    # Balance and shuffle
    positives = positives.shuffle(buffer_size=1000).take(min_samples)
    negatives = negatives.shuffle(buffer_size=1000).take(min_samples)

    # Combine positives and negatives
    balanced_dataset = positives.concatenate(negatives)
    balanced_dataset = balanced_dataset.shuffle(buffer_size=1000)

    return balanced_dataset

    
# Load dataset
dataset, class_weight_dict = load_dataset_with_labels('F:/LizardCV/Raw', None, 10000)
train_dataset = dataset.map(lambda image, label, id: (tf.image.resize(image, [224, 224]), label))
unbatched_ds = train_dataset.unbatch()

# Initialize ensemble
num_classes = 5
ensemble_models = []

for i in range(num_classes):
    print(f"Building model for class {i}")
    model = build_model()
    model.compile(optimizer=Adam(learning_rate=1e-5),
                  loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
                  metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])
    ensemble_models.append(model)

# Loop through and balance dataset for each class
balanced_datasets = []
for i in range(num_classes):
    print(f"Balancing dataset for class {i}")
    balanced_ds = balance_class_dataset(unbatched_ds, class_index=i)
    balanced_datasets.append(balanced_ds.batch(32).prefetch(tf.data.AUTOTUNE))

# Train each model
history_list = []
for i, model in enumerate(ensemble_models):
    print(f"Training model for class {i}")
    binary_dataset = balanced_datasets[i]
    
    # Early stopping callback
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='loss', 
        patience=5, 
        restore_best_weights=True
    )
    
    # Train the model
    history = model.fit(
        binary_dataset,
        epochs=10,
        #class_weight={0: 1, 1: class_weight_dict[i]},  # Class weights for binary classification
        callbacks=[early_stopping_callback]
    )
    history_list.append(history)

# Save models
for i, model in enumerate(ensemble_models):
    model.save(f'F:/LizardCV/ensemble_model_{i}.h5')

# Inference: Ensemble prediction
def ensemble_predict(ensemble_models, dataset):
    predictions = []
    for model in ensemble_models:
        predictions.append(model.predict(dataset, verbose=0))
    predictions = np.concatenate(predictions, axis=1)  # Shape: (num_samples, num_classes)
    print(predictions)
    return np.argmax(predictions, axis=1)  # Class with the highest probability

if ensemble_models is None:
    ensemble_models=[]
    for i in range(5):
        model = tf.keras.models.load_model(f'F:/LizardCV/ensemble_model_{i}.h5')
        model.compile(optimizer=Adam(learning_rate=1e-5),
                  loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
                  metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])
        ensemble_models.append(model)

# Data loading
dataset, class_weight_dict = load_dataset_with_labels('F:/LizardCV/Raw-Test',None,3000)
dataset_for_testing = dataset.map(lambda image, label, id: (tf.image.resize(image, [224, 224]), label))
dataset_for_testing.batch(32)
# Unbatch the dataset to get individual elements
unbatched_ds = dataset_for_testing.unbatch()

# Extract labels after unbatching
labels = []
for image, label in unbatched_ds:
    labels.append(label.numpy())
truth = np.argmax(labels, axis = 1)
print('Labels extracted')

# Get predictions
predictions = ensemble_predict(ensemble_models, dataset_for_testing)
print(classification_report(truth, predictions))

truth_to_predictions = {}
for i in range(len(truth)):
    true_label = truth[i]
    predicted_label = predictions[i]
    if true_label not in truth_to_predictions:
        truth_to_predictions[true_label] = []
    truth_to_predictions[true_label].append(predicted_label)

print("Truth-to-Predictions mapping:")
for true_label, predictions in truth_to_predictions.items():
    print(f"True Label {true_label}: {Counter(predictions)}")
