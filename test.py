import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from utility import *
from keras.optimizers import Adam

# Load the model
model = tf.keras.models.load_model('F:/LizardCV/model.h5')
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy',tf.keras.metrics.Recall(),tf.keras.metrics.Precision()])

# Check the model summary to ensure it's loaded properly
model.summary()

# Data loading
dataset, class_weight_dict = load_dataset_with_labels()
dataset_for_testing = dataset.map(lambda image, label, id: (tf.image.resize(image, [256, 256]), label))
dataset_for_testing.batch(32)

print("Dataset loaded")


# Make predictions on the test data
"""
results = model.evaluate(dataset_for_testing)
print("Test loss:", results[0])
if len(results) > 1:
    for i, metric in enumerate(model.metrics_names[1:], start=1):
        print(f"Test {metric}: {results[i]}")
"""

#Prediction method
# Unbatch the dataset to get individual elements
unbatched_ds = dataset_for_testing.unbatch()

# Extract labels after unbatching
labels = []
for image, label in unbatched_ds:
    labels.append(label.numpy())
truth = np.argmax(labels, axis = 1)
print('Labels extracted')

#test_dataset = test_dataset.map(lambda x, y: x)
predictions = model.predict(dataset_for_testing)
predicted_classes = np.argmax(predictions, axis=1)

# Combine predictions and true labels into a single array for comparison
comparison = list(zip(predicted_classes, labels))

# Display a few examples
for i in range(50):  # Display first 10 comparisons
    
    print(f"Predicted: {predicted_classes[i]}, True Label: {truth[i]}")

# Calculate accuracy
accuracy = accuracy_score(truth, predicted_classes)
print(f"Accuracy: {accuracy}")

# Generate a classification report
print(classification_report(truth, predicted_classes))