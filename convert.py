import os
import tensorflow as tf

# Load the SavedModel
model = tf.saved_model.load('C:/Users/Dallaire/Desktop/LizardsCV/models/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/saved_model')
# Get the inference function
detect_fn = model.signatures['serving_default']

# Define a custom Keras layer to wrap the detection function
class DetectionLayer(tf.keras.layers.Layer):
    def __init__(self, detect_fn):
        super(DetectionLayer, self).__init__()
        self.detect_fn = detect_fn

    def call(self, inputs):
        # Ensure inputs are a proper tensor
        return self.detect_fn(inputs)
    def get_config(self):
        config = super().get_config()
        # Add any additional parameters here if necessary
        return config
# Define Keras Input for the model
inputs = tf.keras.Input(shape=(320, 320, 3), batch_size=None)

# Wrap the detection function in a custom layer
detection_layer = DetectionLayer(detect_fn)(inputs)

# Create a new Keras model
new_model = tf.keras.Model(inputs=inputs, outputs=detection_layer)

# Save the new model in H5 format
tf.keras.models.save_model(new_model, 'C:/Users/Dallaire/Desktop/LizardsCV/models/base.h5')