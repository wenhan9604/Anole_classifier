import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from keras.applications import MobileNetV2
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score, classification_report
from utility import *
from collections import Counter

# Load the MobileNetV2 model pre-trained on ImageNet
base_model = MobileNetV2(input_shape=(256, 256, 3), include_top=False, weights='imagenet')

# Freeze the base model (i.e., make its weights non-trainable)
base_model.trainable = False

# Add custom classification head
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(5, activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy',tf.keras.metrics.Recall(),tf.keras.metrics.Precision()])

# Data loading
dataset, class_weight_dict = load_dataset_with_labels('F:/LizardCV/Raw')
train_dataset = dataset.map(lambda image, label, id: (tf.image.resize(image, [256, 256]), label))

# Train the model
history = model.fit(
    train_dataset,
    epochs=10,
    class_weight=class_weight_dict
)

# Unfreeze some layers of the base model for fine-tuning
base_model.trainable = True
fine_tune_at = 100  # Unfreeze all layers after layer 100

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Recompile the model with a lower learning rate for fine-tuning
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy',tf.keras.metrics.Recall(),tf.keras.metrics.Precision()])

# Continue training (fine-tuning)
history_fine = model.fit(
    train_dataset,
    epochs=10,
    class_weight=class_weight_dict
)

# Save the entire model to a file
model.save('F:/LizardCV/model.h5')

unbatched_ds = train_dataset.unbatch()

# Extract labels after unbatching
labels = []
for image, label in unbatched_ds:
    labels.append(label.numpy())
truth = np.argmax(labels, axis = 1)
print('Labels extracted')

truth_sum = Counter(truth)
print(f'True label counts: {truth_sum}')
