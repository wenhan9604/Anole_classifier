import os
import json
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.losses import MeanAbsoluteError

def load_json_annotations(json_path):
    with open(json_path, 'r') as f:
        annotations = json.load(f)
    return annotations

def parse_annotations(annotation):
    bboxes = []
    
    # Extract image dimensions
    image_width = annotation['imageWidth']
    image_height = annotation['imageHeight']
    
    if annotation['shapes']:
        shape = annotation['shapes'][0]
        points = shape['points']
        # Convert points to (xmin, ymin, xmax, ymax)
        xmin, ymin = points[0]
        xmax, ymax = points[1]
        bboxes.append([xmin, ymin, xmax, ymax])

    # Normalize bounding box coordinates to [0, 1]
    bboxes = np.array(bboxes)
    bboxes[:, [1, 3]] /= image_height  # Normalize ymin, ymax
    bboxes[:, [0, 2]] /= image_width   # Normalize xmin, xmax
    
    return bboxes

def rotate_bboxes(bboxes, angle):
    """Adjust bounding box coordinates based on the rotation angle."""
    rotated_bboxes = []
    for bbox in bboxes:
        xmin, ymin, xmax, ymax = bbox
        if angle == 90:
            # Rotate 90 degrees clockwise
            rotated_bbox = [ymin, 1 - xmax, ymax, 1 - xmin]
        elif angle == 180:
            # Rotate 180 degrees
            rotated_bbox = [1 - xmax, 1 - ymax, 1 - xmin, 1 - ymin]
        elif angle == 270:
            # Rotate 270 degrees clockwise
            rotated_bbox = [1 - ymax, xmin, 1 - ymin, xmax]
        else:
            rotated_bbox = bbox
        rotated_bboxes.append(rotated_bbox)
    return np.array(rotated_bboxes)

def rotate_image(image, angle):
    if angle == 90:
        return tf.image.rot90(image, k=1)
    elif angle == 180:
        return tf.image.rot90(image, k=2)
    elif angle == 270:
        return tf.image.rot90(image, k=3)
    else:
        return image

def load_image_and_labels(image_path, annotation):
    # Load the image
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    
    # Store the original size for later scaling
    original_size = tf.shape(image)[:2]  # Height, Width
    
    # Resize the image to a standard size
    image_resized = tf.image.resize(image, [300, 300])
    
    # Get the corresponding annotations for this image
    bboxes= parse_annotations(annotation)
    #bboxes = np.concatenate([bboxes, c_score.reshape(-1, 1)], axis=-1)
    bboxes = tf.convert_to_tensor(bboxes, dtype=tf.float32)

    # List to hold rotated images and boxes
    images = [image_resized]
    all_bboxes = [bboxes]
    angles = [90, 180]#, 270]
    for angle in angles:
        rotated_image = rotate_image(image_resized, angle)
        rotated_bboxes = rotate_bboxes(bboxes, angle)
        images.append(rotated_image)
        all_bboxes.append(tf.convert_to_tensor(rotated_bboxes, dtype=tf.float32))

    return images, all_bboxes, original_size

def load_dataset(annotations_folder):
    bboxes = []
    images = []
    
    # Iterate through JSON files in the specified folder
    for filename in os.listdir(annotations_folder):
        if filename.endswith('.json'):
            json_path = os.path.join(annotations_folder, filename)
            annotation = load_json_annotations(json_path)
            img_list, bbox_list, origional_size = load_image_and_labels(f'F:/LizardCV/bbox/{annotation["imagePath"]}',annotation)
            
            for img, bbox in zip(img_list, bbox_list):
                images.append(img)
                bboxes.append(bbox)
    
    # Create dataset from image paths and annotations
    dataset = tf.data.Dataset.from_tensor_slices((images, bboxes))
    print(dataset.cardinality().numpy())
    dataset = dataset.batch(8)
    #for batch in dataset.take(1):
    #    print(batch)
    print(dataset.element_spec)
    return dataset

def smooth_l1_loss(y_true, y_pred, delta=1.0):
    loss = tf.where(tf.abs(y_true - y_pred) < delta,
                    0.5 * ((y_true - y_pred) ** 2),
                    delta * tf.abs(y_true - y_pred) - 0.5 * (delta ** 2))
    return tf.reduce_mean(loss)

def custom_loss(y_true, y_pred):
    # Separate bounding boxes and confidence scores
    bbox_true = tf.squeeze(y_true)[:, :4]  # First 4 values: bounding box
    conf_true = tf.squeeze(y_true)[:, 4]  # Last value: confidence score

    bbox_pred = y_pred[:, :4]  # First 4 values: predicted bounding box
    conf_pred = tf.squeeze(y_pred)[:, 4]  # Last value: predicted confidence score

    # Bounding box loss (Mean Squared Error)
    bbox_loss = tf.reduce_mean(tf.square(bbox_true - bbox_pred))

    # Confidence score loss (Binary Cross-Entropy)
    conf_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(conf_true, conf_pred))

    # Total loss is a combination of both
    total_loss = bbox_loss #+ conf_loss
    return total_loss

# Example usage of the dataset loader
annotations_folder = 'F:/LizardCV/bbox'  # Path where JSON files are stored
train_dataset = load_dataset(annotations_folder)
print('Loaded Dataset')
# Load a pre-trained object detection model (SSD MobileNet V2 here as an example)
#base_model = tf.keras.models.load_model('C:/Users/Dallaire/Desktop/LizardsCV/models/base.h5')
base_model = tf.keras.applications.MobileNetV2(input_shape=(300, 300, 3), include_top=False, weights='imagenet')
# Freeze the base model layers to retain pre-trained features
base_model.trainable = False

# Add custom detection layers (you can modify the layers depending on your classes and bounding boxes)
x = base_model.output
x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)  # Detection-specific conv layer
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
output_boxes = tf.keras.layers.Dense(4, activation='sigmoid')(x)  # 4 for bounding box coords

# Create the full model for object detection
detection_model = tf.keras.Model(inputs=base_model.input, outputs=output_boxes)

# Compile the model with appropriate loss functions and an optimizer
detection_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001 / 10), loss=MeanAbsoluteError(), metrics=['mae'])
print('Compiled')

# Train the model (initial training with frozen backbone)
history = detection_model.fit(train_dataset, epochs=10)

# Fine-tuning: Unfreeze some of the layers of the base model for further training
detection_model.trainable = True
fine_tune_at = len(detection_model.layers) // 2  # Unfreeze half of the layers for fine-tuning

for layer in detection_model.layers[:fine_tune_at]:
    layer.trainable = False  # Keep earlier layers frozen

# Compile again with a lower learning rate for fine-tuning
detection_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001 / 10), loss=MeanAbsoluteError(), metrics=['mae'])

# Continue training for fine-tuning
fine_tune_history = detection_model.fit(train_dataset, epochs=10)

# Save the fine-tuned model for inference
detection_model.save('F:/LizardCV/detection.h5')