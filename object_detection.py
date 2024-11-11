import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import random
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_random_file(folder_path):
    # Get a list of all files in the folder
    files = os.listdir(folder_path)
    
    # Filter out directories (if any)
    files = [f for f in files if os.path.isfile(os.path.join(folder_path, f))]
    
    # Choose a random file
    random_file = os.path.join(folder_path,random.choice(files))
    print(random_file)
    return random_file

# Load your custom-trained object detection model
detection_model = tf.keras.models.load_model('F:/LizardCV/detection.h5')

# Load an input image
image = tf.io.read_file(get_random_file('F:/LizardCV/Raw/'))
image = tf.image.decode_jpeg(image, channels=3)  # or tf.image.decode_png if it's a PNG
image = tf.image.resize(image, [300, 300])  # Resize to the input size expected by the model
#image = tf.cast(image, tf.uint8) 
input_tensor = tf.convert_to_tensor(image)
input_tensor = input_tensor[tf.newaxis, ...]  # Add batch dimension
image_np = image.numpy()  # Convert to NumPy array for OpenCV

# Run object detection
detections = detection_model(input_tensor)

# Access the output tensor
output_tensor = detections[0]  # Get the first tensor
output_array = output_tensor.numpy()  # Convert to NumPy array

# Initialize lists to store detected boxes, scores, and class IDs
boxes = []
scores = []

# Set a confidence threshold
confidence_threshold = 0.5

print(output_array)
# Extract bounding box and scores
box = output_array[:4]  # First four elements are the box coordinates
#score = output_array[4]  # The fifth element is the objectness score

# If the score is above the confidence threshold, save the results
#if score >= confidence_threshold:
boxes.append(box)
    #scores.append(score)

# Convert lists to NumPy arrays for easier manipulation
boxes = np.array(boxes)
#scores = np.array(scores)

# Define a color map for visualization
colors = np.random.randint(0, 255, size=(len(boxes), 3), dtype='uint8')

# Draw bounding boxes on the original image
for i in range(len(boxes)):
    xmin, ymin, xmax, ymax = boxes[i]

    # Convert to original image scale
    xmin = int((xmin) * image.shape[1])
    xmax = int((xmax) * image.shape[1])
    ymin = int((ymin) * image.shape[0])
    ymax = int((ymax) * image.shape[0])
    
    # Draw bounding box and label
    cv2.rectangle(image_np, (xmin, ymin), (xmax, ymax), color=colors[i].tolist(), thickness=2)
    #cv2.putText(image_np, f' {scores[i]:.2f}', (xmin, ymin - 10),
    #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i].tolist(), 2)


# Show the image with detections
image_np = image_np.astype(np.uint8)
plt.imshow(image_np)
plt.axis('off')
plt.show()