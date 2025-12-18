import cv2
import numpy as np
import joblib
from skimage.feature import hog
import os

def predict(dataFilePath, bestModelPath):
    model_data = joblib.load(bestModelPath)
    
    svm_model = model_data['svm_model']
    scaler = model_data['scaler']
    feature_type = model_data['feature_type']
    confidence_threshold = model_data['confidence_threshold']
    
    image_files = sorted([
        f for f in os.listdir(dataFilePath)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))
    ])
    
    if len(image_files) == 0:
        return []
    
    def get_color_features(image):
        image = cv2.resize(image, (128, 128))
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0,1,2], None, [8,8,8], [0,180,0,256,0,256])
        cv2.normalize(hist, hist)
        return hist.flatten()
    
    def get_hog_features(image):
        image = cv2.resize(image, (128, 128))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return hog(gray, orientations=9, pixels_per_cell=(8,8),
                   cells_per_block=(2,2), block_norm='L2-Hys')
    
    def extract_features(image):
        if feature_type == 'color':
            return get_color_features(image)
        elif feature_type == 'hog':
            return get_hog_features(image)
        elif feature_type == 'both':
            return np.concatenate([
                get_color_features(image),
                get_hog_features(image)
            ])
        else:
            raise ValueError("Unknown feature type")
    
    predictions = []
    
    for img_file in image_files:
        img_path = os.path.join(dataFilePath, img_file)
        image = cv2.imread(img_path)
        
        if image is None:
            predictions.append(6)
            continue
        
        features = extract_features(image).reshape(1, -1)
        features = scaler.transform(features)
        
        probs = svm_model.predict_proba(features)[0]
        if np.max(probs) < confidence_threshold:
            predictions.append(6)
        else:
            predictions.append(int(np.argmax(probs)))
    
    return predictions
