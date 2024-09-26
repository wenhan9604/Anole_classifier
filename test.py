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
train_dataset, test_dataset = train_test_split(dataset,0.8)

print("Dataset loaded")

# Print model input shape
print(model.input_shape)
for inputs, _ in test_dataset.take(1):
    print(inputs.shape)

# Unbatch the dataset to get individual elements
#unbatched_ds = test_dataset.unbatch()

# Extract labels after unbatching
#labels = []
#for image, label in unbatched_ds:
#    labels.append(label.numpy())
#print('Labels extracted')

# Make predictions on the test data
results = model.evaluate(test_dataset)
print("Test loss:", results[0])
if len(results) > 1:
    for i, metric in enumerate(model.metrics_names[1:], start=1):
        print(f"Test {metric}: {results[i]}")
#test_dataset = test_dataset.map(lambda x, y: x)
#predictions = model.predict(test_dataset_batched)
#predicted_classes = np.argmax(predictions, axis=1)

# Combine predictions and true labels into a single array for comparison
#comparison = list(zip(predicted_classes, labels))

# Display a few examples
#for i in range(10):  # Display first 10 comparisons
#    print(f"Predicted: {predicted_classes[i]}, True Label: {labels[i]}")

# Calculate accuracy
#accuracy = accuracy_score(labels, predicted_classes)
#print(f"Accuracy: {accuracy}")

# Generate a classification report
#print(classification_report(labels, predicted_classes))