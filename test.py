import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split

IMG_HEIGHT = 256
IMG_WIDTH = 256
BATCH_SIZE = 16
EPOCHS = 20
DATA_DIR = "dataset" 

def load_data(data_dir):
    images = []
    masks = []
    img_dir = os.path.join(data_dir, "images")
    mask_dir = os.path.join(data_dir, "masks")
    
    for img_name in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_name)
        mask_path = os.path.join(mask_dir, img_name.replace(".jpg", "_mask.png"))
        
        img = cv2.imread(img_path)
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        img = img / 255.0
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (IMG_WIDTH, IMG_HEIGHT))
        mask = mask / 255.0 
        mask = np.expand_dims(mask, axis=-1)
        
        images.append(img)
        masks.append(mask)
    
    return np.array(images), np.array(masks)

def unet_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)):
    inputs = layers.Input(input_shape)
    
    c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    
    c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    
    c3 = layers.Conv2D(256, 3, activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, 3, activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)
    
    c4 = layers.Conv2D(512, 3, activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(512, 3, activation='relu', padding='same')(c4)
    
    u5 = layers.Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(c4)
    u5 = layers.concatenate([u5, c3])
    c5 = layers.Conv2D(256, 3, activation='relu', padding='same')(u5)
    c5 = layers.Conv2D(256, 3, activation='relu', padding='same')(c5)
    
    u6 = layers.Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c2])
    c6 = layers.Conv2D(128, 3, activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(128, 3, activation='relu', padding='same')(c6)
    
    u7 = layers.Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c1])
    c7 = layers.Conv2D(64, 3, activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(64, 3, activation='relu', padding='same')(c7)
    
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c7)
    
    model = models.Model(inputs, outputs)
    return model

def main():
    images, masks = load_data(DATA_DIR)
    X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)
    
    model = unet_model()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train model
    model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_val, y_val))
    
    # Save model
    model.save("trash_segmentation_model.h5")
    
    # Example inference
    test_img = X_val[0:1]
    pred_mask = model.predict(test_img)
    pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255
    
    # Save or display result
    cv2.imwrite("predicted_mask.png", pred_mask[0])

if __name__ == "__main__":
    main()
