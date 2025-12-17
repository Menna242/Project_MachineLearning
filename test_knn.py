import cv2
import numpy as np
import sys
import os
from knn_train import KNNClassifierWithRejection, CombinedKNNWithRejection
from skimage.feature import hog

def get_color_features(image):
    image = cv2.resize(image, (128, 128))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    color_histogram = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(color_histogram, color_histogram)
    return color_histogram.flatten()

def get_hog_features(image):
    image = cv2.resize(image, (128, 128))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return hog(gray, orientations=9, pixels_per_cell=(8, 8),
               cells_per_block=(2, 2), block_norm='L2-Hys')

def extract_features(image, feature_type='both'):
    if feature_type == 'color':
        return get_color_features(image)
    elif feature_type == 'hog':
        return get_hog_features(image)
    else:
        return np.concatenate([get_color_features(image), get_hog_features(image)])

def test_single_image(image_path, model_type='color', feature_type='both', model_path=None):
    if model_type == 'combined':
        knn = CombinedKNNWithRejection(k=3)
        knn.color_knn = KNNClassifierWithRejection.load_model(model_path['color'])
        knn.hog_knn   = KNNClassifierWithRejection.load_model(model_path['hog'])
        if knn.threshold is None:
            knn.threshold = 0.65*knn.color_knn.rejection_threshold + 0.35*knn.hog_knn.rejection_threshold
    else:
        knn = KNNClassifierWithRejection.load_model(model_path)

    image = cv2.imread(image_path)
    if image is None:
        print(f" Could not load image: {image_path}")
        return

    features = extract_features(image, feature_type).reshape(1, -1)


    if model_type == 'combined':
        X_color = extract_features(image, 'color').reshape(1, -1)
        X_hog   = extract_features(image, 'hog').reshape(1, -1)
        preds, rejected = knn.predict_with_rejection(X_color, X_hog)
        predicted_class = preds[0]
        is_rejected = rejected[0]
    else:
        preds, rejected, _ = knn.predict_with_rejection(features)
        predicted_class = preds[0]
        is_rejected = rejected[0]


    if model_type == 'combined':
        if is_rejected or predicted_class == knn.unknown_class_id:
            print(f"️  Class: UNKNOWN (ID 6)")
        else:
            # Use color_knn class_names for naming
            print(f"✓ Class: {knn.color_knn.class_names[predicted_class].upper()} (ID {predicted_class})")
    else:
        if is_rejected or predicted_class == knn.unknown_class_id:
            print(f"️  Class: UNKNOWN (ID 6)")
        else:
            print(f" Class: {knn.class_names[predicted_class].upper()} (ID {predicted_class})")


def test_folder(folder_path, model_type='color', feature_type='both', model_path=None):
    """Test all images in a folder"""
    image_files = [f for f in os.listdir(folder_path)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    if not image_files:
        print(" No images found in folder")
        return

    print(f"\nFound {len(image_files)} images. Processing...\n")
    for i, img_file in enumerate(image_files, 1):
        img_path = os.path.join(folder_path, img_file)
        test_single_image(img_path, model_type=model_type, feature_type=feature_type, model_path=model_path)
        if i % 10 == 0:
            print(f"Processed {i}/{len(image_files)} images")



if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "\nUsage:\n  python test_knn.py <image_path> --model <model_path>\n"
            "  python test_knn.py <folder_path> --folder --model <model_path>\n"
            "  python test_knn.py --realtime --model <model_path>"
        )
        sys.exit(1)


    feature_type = 'both'
    model_type = 'color'
    model_path = None

    if '--model' in sys.argv:
        idx = sys.argv.index('--model')
        model_path = sys.argv[idx + 1]

    if model_path is None:
        raise ValueError("You must provide a trained model path for testing")

    if '--features' in sys.argv:
        idx = sys.argv.index('--features')
        feature_type = sys.argv[idx + 1]

    if '--combined' in sys.argv:
        model_type = 'combined'


    if '--model' in sys.argv:
        idx = sys.argv.index('--model')
        path_arg = sys.argv[idx + 1]
        if model_type == 'combined':
            # For combined, expect two separate pickle files
            model_path = {'color': path_arg + '_color.pkl', 'hog': path_arg + '_hog.pkl'}
        else:
            model_path = path_arg
    else:
        raise ValueError("You must provide a trained model path for testing")


    if '--folder' in sys.argv:
        test_folder(sys.argv[1], model_type=model_type, feature_type=feature_type, model_path=model_path)
    else:
        test_single_image(sys.argv[1], model_type=model_type, feature_type=feature_type, model_path=model_path)
