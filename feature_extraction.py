# feature_extraction.py
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

def get_feature_extractor(model_path):
    model = tf.keras.models.load_model(model_path)
    feature_extractor = tf.keras.Model(inputs=model.input, outputs=model.get_layer('dense').output)
    return feature_extractor

def extract_features(img_path, feature_extractor):
    img = image.load_img(img_path, target_size=(128, 128))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = tf.keras.applications.mobilenet_v2.preprocess_input(img_data)
    features = feature_extractor.predict(img_data)
    return features
