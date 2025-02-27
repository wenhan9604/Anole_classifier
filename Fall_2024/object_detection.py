import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import random
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import keras_cv

"""
Selects a random file to test Object detetion on.
"""

def get_random_file(folder_path):
    # Get a list of all files in the folder
    files = os.listdir(folder_path)
    
    # Filter out directories (if any)
    files = [f for f in files if os.path.isfile(os.path.join(folder_path, f))]
    
    # Choose a random file
    random_file = os.path.join(folder_path,random.choice(files))
    print(random_file)
    return random_file

class CustomCIoULoss(keras_cv.losses.CIoULoss):
    @classmethod
    def from_config(cls, config):
        config['bounding_box_format'] = "xyxy"  # Ensure format is set
        return super().from_config(config)

detection_model = tf.keras.models.load_model(
    'F:/LizardCV/detection.h5',
    custom_objects={
        "YOLOV8Detector": keras_cv.models.YOLOV8Detector,
    },
    compile=False,  # Prevent loss compilation
)

ciou_loss = keras_cv.losses.CIoULoss(bounding_box_format="xyxy")
classification_loss = tf.keras.losses.BinaryCrossentropy()

detection_model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    box_loss=ciou_loss,
    classification_loss=classification_loss,
    metrics=None
)

# Load an input image
image = tf.io.read_file(get_random_file('F:/LizardCV/Raw/'))
image = tf.image.decode_jpeg(image, channels=3)  # or tf.image.decode_png if it's a PNG
image = tf.image.resize(image, [320, 320])  # Resize to the input size expected by the model
#image = tf.cast(image, tf.uint8) 
input_tensor = tf.convert_to_tensor(image)
input_tensor = input_tensor[tf.newaxis, ...]  # Add batch dimension
image_np = image.numpy()  # Convert to NumPy array for OpenCV

# Run object detection
detections = detection_model.predict(input_tensor)
print(detections) #This is giving shape=(1, 2100, 64) for boxes. Something is wrong with that.
# Extract and preprocess detections
boxes = detections['boxes'].numpy()[0]  # Remove batch dimension
classes = detections['classes'].numpy()[0]  # Remove batch dimension

# Apply a confidence threshold (optional)
confidence_threshold = 0.5
filtered_indices = boxes[0, 4] > confidence_threshold
filtered_boxes = boxes[filtered_indices]  # Shape: (num_filtered, 4)
filtered_classes = classes[filtered_indices]  # Shape: (num_filtered, 1)

# Assuming input_tensor.shape is (1, height, width, 3)
input_height, input_width = input_tensor.shape[1:3]

# Denormalize box coordinates
filtered_boxes[:, 0] *= input_width  # x_min
filtered_boxes[:, 1] *= input_height  # y_min
filtered_boxes[:, 2] *= input_width  # x_max
filtered_boxes[:, 3] *= input_height  # y_max

# Define a color map for visualization
colors = np.random.randint(0, 255, size=(len(boxes), 3), dtype='uint8')

# Draw the bounding boxes
for box in filtered_boxes[:,:4]:
    x_min, y_min, x_max, y_max = box
    rect = plt.Rectangle(
        (x_min, y_min),
        x_max - x_min,
        y_max - y_min,
        fill=False,
        color='red',
        linewidth=2
    )
    plt.gca().add_patch(rect)

# Show the image with detections
image_np = image_np.astype(np.uint8)
plt.imshow(image_np)
plt.axis('off')
plt.show()