import cv2
import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input

def preprocess_rf_svm(img_path, target_size=(32, 32)):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert BGR to RGB
    img = cv2.resize(img, target_size)          # resize to target_size
    img_flat = img.flatten() / 255.0            # Flatten and normalize to [0, 1]
    return img_flat

def preprocess_vgg16(img_path, target_size=(224, 224)):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    img = cv2.resize(img, target_size)         
    img_array = np.array(img).astype(np.float32)
    img_array = preprocess_input(img_array)     
    return img_array