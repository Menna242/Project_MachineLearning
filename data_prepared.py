
import cv2
import numpy as np
import os

def get_image_features(image):    
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



features=[]
labels=[]

classes = [ "cardboard","glass","metal", "paper", "plastic", "trash"]

for name in classes:
    pathfolder = os.path.join("dataset",name)
    imgs = os.listdir(pathfolder)
    for img_name in imgs:
        img_path = os.path.join(pathfolder, img_name)
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        
        # img_path = os.path.join(pathfolder, img_name)
        image = cv2.imread(img_path)

        if image is None:
            # print(f" {img_path} ")
            continue
        img_features = get_image_features(image)
        features.append(img_features)  
        labels.append(classes.index(name))

        augmented_images = image_augmentation(image)  
        for aug_img in augmented_images:
            aug_features = get_image_features(aug_img)
            features.append(aug_features)
            labels.append(classes.index(name))


features = np.array(features)
labels = np.array(labels)

np.save('features.npy', features)
np.save('labels.npy', labels)


# print("features labels")
# for i in range(5): # first 5 images in label 0 and it is a precent of number of pixels in tis color h0s0v0,h0s0v1...
#     print(f"Feature vector  {i} (label={labels[i]}):")
#     print(features[i])
#     print("-"*60)



        
