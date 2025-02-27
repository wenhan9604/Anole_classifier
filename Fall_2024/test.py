import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from utility import *
from keras.optimizers import Adam
from collections import Counter

"""
Testing for the one-hot encoded model.
"""

# Load object detection model
detection_model = tf.keras.models.load_model('F:/LizardCV/detection.h5')

# Load the model
model = tf.keras.models.load_model('F:/LizardCV/test_model.h5')
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy',tf.keras.metrics.Recall(),tf.keras.metrics.Precision()])

# Check the model summary to ensure it's loaded properly
model.summary()

# Data loading
"""
Second argument is if object detection is used to crop the image. Third argument is the maximum size of each class loaded into the dataset
"""
dataset, class_weight_dict = load_dataset_with_labels('F:/LizardCV/Raw-Test',None,3000)
dataset_for_testing = dataset.map(lambda image, label, id: (tf.image.resize(image, [224, 224]), label))
dataset_for_testing.batch(32)

print("Dataset loaded")

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
top_two = np.argsort(predictions,axis=1)[:,-2:]

# Combine predictions and true labels into a single array for comparison
comparison = list(zip(predicted_classes, labels))

# Display a few examples
predicted_classes_sum = Counter(predicted_classes)
print(f'Prediction label counts: {predicted_classes_sum}')
truth_sum = Counter(truth)
print(f'True label counts: {truth_sum}')

# Calculate accuracy
accuracy = accuracy_score(truth, predicted_classes)
print(f"Accuracy: {accuracy}")
correct_top_two = [1 if truth[i] in top_two[i] else 0 for i in range(len(truth))]
top_two_acc = np.mean(correct_top_two)
print(f'Top 2 acc: {top_two_acc}')

truth_to_predictions = {}
for i in range(len(truth)):
    true_label = truth[i]
    predicted_label = predicted_classes[i]
    if true_label not in truth_to_predictions:
        truth_to_predictions[true_label] = []
    truth_to_predictions[true_label].append(predicted_label)

print("Truth-to-Predictions mapping:")
for true_label, predictions in truth_to_predictions.items():
    print(f"True Label {true_label}: {Counter(predictions)}")


# Generate a classification report
print(classification_report(truth, predicted_classes))