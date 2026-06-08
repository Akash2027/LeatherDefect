import cv2
import os
import numpy as np
import albumentations as A
from glob import glob

# Input and output directories
input_dir = "LEATHERDATASETv1"  # Folder containing original images
output_dir = "AUGMENTED_IMAGES"  # Folder where cropped & augmented images are saved

# Create output directory if not exists
os.makedirs(output_dir, exist_ok=True)

# Check if input images exist
image_files = glob(os.path.join(input_dir, "*.jpg")) + glob(os.path.join(input_dir, "*.png"))
print(f"Total images found: {len(image_files)}")  # Debug print

if len(image_files) == 0:
    print("⚠ No images found! Check input directory path.")
    exit()

# Define Albumentations augmentation pipeline
augmentations = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.Rotate(limit=20, p=0.5),
    A.GaussianBlur(p=0.2),
    A.RandomCrop(width=640, height=640, p=1.0)  # Ensures a 640x640 crop
])

# Number of augmentations per image
num_augmentations = 50

for img_path in image_files:
    # Read image
    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠ Skipping {img_path} (could not load).")
        continue

    # Get filename without extension
    base_filename = os.path.splitext(os.path.basename(img_path))[0]

    # Ensure image is at least 640x640
    if img.shape[0] < 640 or img.shape[1] < 640:
        print(f"⚠ Skipping {img_path} (too small).")
        continue

    print(f"🔄 Processing: {img_path}, Size: {img.shape}")

    # Generate multiple augmented images
    for i in range(1, num_augmentations + 1):
        augmented = augmentations(image=img)["image"]

        # Check augmented image shape
        print(f"Augmented shape: {augmented.shape}")

        # Save the augmented image with a new filename
        new_filename = f"{base_filename}_{i:02d}.jpg"
        new_filepath = os.path.join(output_dir, new_filename)

        # Save the image and verify success
        success = cv2.imwrite(new_filepath, augmented)
        if success:
            print(f"✅ Saved: {new_filepath}")
        else:
            print(f"❌ Failed to save: {new_filepath}")

print("🚀 Augmentation and cropping completed successfully!")
