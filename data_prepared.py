import cv2
import numpy as np
import os
from skimage.feature import hog

def get_color_features(image):
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





def get_hog_features(image):
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


def image_augmentation(image):
    img_augmentation = []
    
    #Brightness
    for factor in [0.7, 1.3]:  
        bright = cv2.convertScaleAbs(image, alpha=factor, beta=0) 
        img_augmentation.append(bright)


    #Rotation
    for angle in [-30,30]:
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        rotated = cv2.warpAffine(image, M, (w, h))
        img_augmentation.append(rotated) # 2 image 
    
    #flip
    img_augmentation.append(cv2.flip(image,0))
    return img_augmentation


# ---- Data structures for all three cases ----
features_color = []
features_hog = []
features_both = []
labels = []

classes = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

for name in classes:
    pathfolder = os.path.join("dataset", name)
    imgs = os.listdir(pathfolder)
    for img_name in imgs:
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img_path = os.path.join(pathfolder, img_name)
        image = cv2.imread(img_path)
        if image is None:
            continue

        # ---- Original Image ----
        color_feat = get_color_features(image)
        hog_feat = get_hog_features(image)

        features_color.append(color_feat)
        features_hog.append(hog_feat)
        features_both.append(np.concatenate([color_feat, hog_feat]))
        labels.append(classes.index(name))

        # ---- Augmented Images ----
        augmented_images = image_augmentation(image)
        for aug_img in augmented_images:
            color_feat = get_color_features(aug_img)
            hog_feat = get_hog_features(aug_img)

            features_color.append(color_feat)
            features_hog.append(hog_feat)
            features_both.append(np.concatenate([color_feat, hog_feat]))
            labels.append(classes.index(name))

# ---- Convert to numpy arrays ----
features_color = np.array(features_color)
features_hog = np.array(features_hog)
features_both = np.array(features_both)
labels = np.array(labels)

# ---- Save to files ----
np.save('features_color.npy', features_color)
np.save('features_hog.npy', features_hog)
np.save('features_both.npy', features_both)
np.save('labels.npy', labels)

print("Saved features:")
print("Color only length:", features_color.shape[1])
print("HOG only length:", features_hog.shape[1])
print("Color + HOG length:", features_both.shape[1])


# print("features labels")
# for i in range(5): # first 5 images in label 0 and it is a precent of number of pixels in tis color h0s0v0,h0s0v1...
#     print(f"Feature vector  {i} (label={labels[i]}):")
#     print(features[i])
#     print("-"*60)



        
