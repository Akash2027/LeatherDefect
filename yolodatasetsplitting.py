import os
import shutil
import random

# Define dataset paths (Keep as it is)
dataset_root = r"C:\Users\AdminUser\Desktop\WINTER SEM 24-25\LeatherDatasetPreprocessing\ROUND2\INPUTS\LARVA_DATASET"
output_root = r"C:\Users\AdminUser\Desktop\WINTER SEM 24-25\LeatherDatasetPreprocessing\ROUND2\OUTPUTS\LARVA_YOL"

# Define subfolders
source_images = os.path.join(dataset_root, "images")  # All images are here
source_labels = os.path.join(dataset_root, "labels")  # Annotations are here

# Create YOLO dataset structure
os.makedirs(f"{output_root}/images/train", exist_ok=True)
os.makedirs(f"{output_root}/images/val", exist_ok=True)
os.makedirs(f"{output_root}/images/test", exist_ok=True)
os.makedirs(f"{output_root}/labels/train", exist_ok=True)
os.makedirs(f"{output_root}/labels/val", exist_ok=True)
os.makedirs(f"{output_root}/labels/test", exist_ok=True)

# Get all image filenames
all_images = [f for f in os.listdir(source_images) if f.endswith(('.jpg', '.png'))]
random.shuffle(all_images)

# Split dataset (80% train, 10% val, 10% test)
train_split = int(0.8 * len(all_images))
val_split = int(0.9 * len(all_images))

train_images = all_images[:train_split]
val_images = all_images[train_split:val_split]
test_images = all_images[val_split:]


def move_files(image_list, subset, source_img_folder, source_lbl_folder):
    for img in image_list:
        # Move images
        shutil.copy(os.path.join(source_img_folder, img), os.path.join(output_root, f"images/{subset}/{img}"))

        # Move labels (if exists)
        label_file = os.path.splitext(img)[0] + '.txt'
        label_path = os.path.join(source_lbl_folder, label_file)
        if os.path.exists(label_path):
            shutil.copy(label_path, os.path.join(output_root, f"labels/{subset}/{label_file}"))
        else:
            # If no annotation exists, create an empty label file (required for YOLO training)
            open(os.path.join(output_root, f"labels/{subset}/{label_file}"), 'w').close()


# Move files to YOLO dataset structure
move_files(train_images, "train", source_images, source_labels)
move_files(val_images, "val", source_images, source_labels)
move_files(test_images, "test", source_images, source_labels)

print("YOLO dataset successfully created at:", output_root)
