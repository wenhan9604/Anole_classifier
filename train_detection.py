import os
import json
import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt
from tensorflow.keras.losses import Huber
import keras_cv

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
    #Adjust bounding box coordinates based on the rotation angle.
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
    image_resized = tf.image.resize(image, [320, 320])
    
    # Get the corresponding annotations for this image
    bboxes= parse_annotations(annotation)
    #bboxes = np.concatenate([bboxes, c_score.reshape(-1, 1)], axis=-1)
    #bboxes = tf.ragged.constant(bboxes, dtype=tf.float32)
    classes = tf.convert_to_tensor(tf.zeros((tf.shape(bboxes)[0]), dtype=tf.float32))

    # List to hold rotated images and boxes
    outputs = []
    angles = [90, 180]#, 270]
    outputs.append((image_resized, {"bbox": bboxes, "classes": classes}))
    for angle in angles:
        rotated_image = rotate_image(image_resized, angle)
        rotated_bboxes = rotate_bboxes(bboxes, angle)
        outputs.append((rotated_image,{"bbox": rotated_bboxes, "classes": classes}))

    return outputs

def load_dataset(annotations_folder):
    images = []
    bboxes = []
    classes = []
    dataset = []
    bounding_boxes = {"boxes": [], "classes": []}
    # Iterate through JSON files in the specified folder
    for filename in os.listdir(annotations_folder):
        if filename.endswith('.json'):
            json_path = os.path.join(annotations_folder, filename)
            annotation = load_json_annotations(json_path)
            outputs = load_image_and_labels(f'C:/Lizards/bbox/{annotation["imagePath"]}',annotation)
            for row in outputs:
                images.append(row[0])
                bboxes.append(row[1]["bbox"])
                classes.append(row[1]["classes"])
    #images_tensor = tf.convert_to_tensor(images)

    # Create dataset from image paths and annotations
    images = tf.stack(images)
    bounding_boxes["boxes"] = bboxes
    bounding_boxes["classes"] = classes
    # Return a tf.data.Dataset
    dataset = tf.data.Dataset.from_tensor_slices({
        "images": images,
        "bounding_boxes": bounding_boxes
    })
   
    dataset = dataset.cache().shuffle(1000).batch(4).prefetch(tf.data.AUTOTUNE)

    for batch in dataset.take(1):
        print("Images shape:", batch["images"].shape)
        print("Boxes shape:", batch["bounding_boxes"]["boxes"].shape)
        print("Classes shape:", batch["bounding_boxes"]["classes"].shape)
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
annotations_folder = 'C:/Lizards/bbox'  # Path where JSON files are stored
train_dataset = load_dataset(annotations_folder)
print('Loaded Dataset')

backbone = keras_cv.models.YOLOV8Backbone.from_preset(
    "yolo_v8_m_backbone_coco",
    load_weights=True
)
 
yolo = keras_cv.models.YOLOV8Detector(
    num_classes=1,
    bounding_box_format="xyxy",
    backbone=backbone,
    fpn_depth=3,
)
 
yolo.summary()

optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.001,
    global_clipnorm=10.0,
)
 
yolo.compile(
    optimizer=optimizer, classification_loss="binary_crossentropy", box_loss="ciou"
) 
	
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs_yolov8large")

history = yolo.fit(
    train_dataset,
    epochs=10
)

yolo.save('C:/Lizards/Anole_Classifier/detection.h5')