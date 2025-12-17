import cv2
import numpy as np
import joblib
from skimage.feature import hog
import os

class SVMTest:
    def __init__(self, model_path):
        print(f"Loading model from {model_path}...")
        model_data = joblib.load(model_path)
        
        self.svm_model = model_data['svm_model']
        self.scaler = model_data['scaler']
        self.feature_type = model_data['feature_type']
        self.confidence_threshold = model_data['confidence_threshold']
        self.class_names = model_data['class_names'] + ['Unknown']
        
        print(f"Model loaded successfully")
        print(f"Feature type: {self.feature_type}")
        print(f"Confidence threshold: {self.confidence_threshold}")
    
    def get_color_features(self, image):
        image = cv2.resize(image, (128, 128))
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
        color_histogram = cv2.calcHist(
            [hsv],
            [0, 1, 2],
            None,
            [8, 8, 8],
            [0, 180, 0, 256, 0, 256]
        )
        
        cv2.normalize(color_histogram, color_histogram)
        return color_histogram.flatten()
    
    def get_hog_features(self, image):
        image = cv2.resize(image, (128, 128))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        hog_features = hog(
            gray,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm='L2-Hys'
        )
        return hog_features
    
    def extract_features(self, image):
        if self.feature_type == 'color':
            return self.get_color_features(image)
        elif self.feature_type == 'hog':
            return self.get_hog_features(image)
        elif self.feature_type == 'both':
            color_feat = self.get_color_features(image)
            hog_feat = self.get_hog_features(image)
            return np.concatenate([color_feat, hog_feat])
        else:
            raise ValueError(f"Unknown feature type: {self.feature_type}")
    
    def predict(self, image):
        features = self.extract_features(image)
        features = features.reshape(1, -1)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        probabilities = self.svm_model.predict_proba(features_scaled)[0]
        max_confidence = np.max(probabilities)
        predicted_class = np.argmax(probabilities)
      
        if max_confidence < self.confidence_threshold:
            final_class = 6  
            final_name = 'Unknown'
        else:
            final_class = predicted_class
            final_name = self.class_names[predicted_class]
        
        return {
            'class_id': int(final_class),
            'class_name': final_name,
            'confidence': float(max_confidence),
            'probabilities': {self.class_names[i]: float(probabilities[i]) 
                            for i in range(len(probabilities))}
        }
    
    def predict_batch(self, images):
        results = []
        for img in images:
            results.append(self.predict(img))
        return results


def test_on_single_image(model_path, image_path):
    print(f"\n{'='*60}")
    print("TESTING ON SINGLE IMAGE")
    print(f"{'='*60}")
    
    # Load model
    predictor = SVMTest(model_path)
    
    # Load image
    print(f"\nLoading image: {image_path}")
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    print(f"Image shape: {image.shape}")
    
    # Predict
    result = predictor.predict(image)
    
    # Display results
    print(f"\n{'='*60}")
    print("PREDICTION RESULTS")
    print(f"{'='*60}")
    print(f"Predicted Class: {result['class_name']} (ID: {result['class_id']})")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"\nAll Class Probabilities:")
    for class_name, prob in result['probabilities'].items():
        print(f"  {class_name:12s}: {prob:.4f}")
    


def test_on_folder(model_path, folder_path):
    print(f"\n{'='*60}")
    print("TESTING ON FOLDER")
    print(f"{'='*60}")
    
    # Load model
    predictor = SVMTest(model_path)
    
    # Get all image files
    image_files = [f for f in os.listdir(folder_path) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"\nFound {len(image_files)} images in {folder_path}")
    
    results = []
    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)
        image = cv2.imread(img_path)
        
        if image is None:
            continue
        
        result = predictor.predict(image)
        result['filename'] = img_file
        results.append(result)
        
        print(f"\n{img_file}:")
        print(f"  Predicted: {result['class_name']} (confidence: {result['confidence']:.4f})")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    class_counts = {}
    for result in results:
        class_name = result['class_name']
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    print("\nPrediction distribution:")
    for class_name, count in sorted(class_counts.items()):
        print(f"  {class_name:12s}: {count} images")
    
    avg_confidence = np.mean([r['confidence'] for r in results])
    print(f"\nAverage confidence: {avg_confidence:.4f}")



def main():
    print("\n" + "="*60)
    print("SVM CLASSIFIER - TESTING INTERFACE")
    print("="*60)
    
    print("\nAvailable test modes:")
    print("1. Test on single image")
    print("2. Test on folder of images")
    
    choice = input("\nEnter your choice (1/2): ").strip()
    
    # Ask for model path
    print("\nAvailable models:")
    print("  - svm_model_color.pkl")
    print("  - svm_model_hog.pkl")
    print("  - svm_model_both.pkl")
    
    model_path = input("\nEnter model path (or press Enter for 'svm_model_both.pkl'): ").strip()
    if not model_path:
        model_path = 'svm_model_both.pkl'
    
    if not os.path.exists(model_path):
        print(f"\nError: Model file '{model_path}' not found!")
        return
    
    if choice == '1':
        image_path = input("\nEnter image path: ").strip()
        test_on_single_image(model_path, image_path)
    
    elif choice == '2':
        folder_path = input("\nEnter folder path: ").strip()
        test_on_folder(model_path, folder_path)
    else:
        print("Invalid choice!")

    print("\nTesting finished. Exiting program...")
    import sys
    sys.exit(0)


if __name__ == "__main__":
    main()