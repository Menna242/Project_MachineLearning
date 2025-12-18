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
    
    image_files = sorted([f for f in os.listdir(dataFilePath) 
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp','.webp'))])
    
    if len(image_files) == 0:
        print(f"Warning: No images found in {dataFilePath}")
        return []
    
    def get_color_features(image):
        image_resized = cv2.resize(image, (128, 128))
        hsv = cv2.cvtColor(image_resized, cv2.COLOR_BGR2HSV)
        
        color_histogram = cv2.calcHist(
            [hsv],
            [0, 1, 2],
            None,
            [8, 8, 8],
            [0, 180, 0, 256, 0, 256]
        )
        
        cv2.normalize(color_histogram, color_histogram)
        return color_histogram.flatten()
    
    def get_hog_features(image):
        image_resized = cv2.resize(image, (128, 128))
        gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
        
        hog_features = hog(
            gray,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm='L2-Hys'
        )
        return hog_features
    
    def extract_features(image):
        if feature_type == 'color':
            return get_color_features(image)
        elif feature_type == 'hog':
            return get_hog_features(image)
        elif feature_type == 'both':
            color_feat = get_color_features(image)
            hog_feat = get_hog_features(image)
            return np.concatenate([color_feat, hog_feat])
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
        
    predictions = []
    
    for i, img_file in enumerate(image_files, 1):
        img_path = os.path.join(dataFilePath, img_file)
        image = cv2.imread(img_path)
        
        if image is None:
            print(f"Warning: Could not load {img_file}, predicting Unknown (6)")
            predictions.append(6)
            continue
        
        features = extract_features(image)
        features = features.reshape(1, -1)
        features_scaled = scaler.transform(features)
        
        probabilities = svm_model.predict_proba(features_scaled)[0]
        max_confidence = np.max(probabilities)
        predicted_class = np.argmax(probabilities)
        
        if max_confidence < confidence_threshold:
            final_prediction = 6  # Unknown class
        else:
            final_prediction = predicted_class
        
        predictions.append(final_prediction)
        
    predictions = [int(p) for p in predictions]

    print(f"\nPrediction complete!")
    print(f"Total predictions: {len(predictions)}")
    
    unique, counts = np.unique(predictions, return_counts=True)
    print(f"\nPrediction distribution:")
    class_names = ['Glass', 'Paper', 'Cardboard', 'Plastic', 'Metal', 'Trash', 'Unknown']
    for cls, count in zip(unique, counts):
        print(f"  Class {cls} ({class_names[cls]}): {count} images")
    
    return predictions
