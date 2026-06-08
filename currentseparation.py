import os
import shutil

# Define paths
images_folder = r"C:\Users\AdminUser\Desktop\WINTER SEM 24-25\LeatherDatasetPreprocessing\Newdefects\Image"  # Update this path
annotations_folder = r"C:\Users\AdminUser\Desktop\WINTER SEM 24-25\LeatherDatasetPreprocessing\Newdefects\Annotation"  # Update this path
output_images_folder = r"C:\Users\AdminUser\Desktop\WINTER SEM 24-25\LeatherDatasetPreprocessing\YOLODATASETv3\images"  # Update this path
output_annotations_folder = r"C:\Users\AdminUser\Desktop\WINTER SEM 24-25\LeatherDatasetPreprocessing\YOLODATASETv3\labels"  # Update this path

# Ensure output directories exist
os.makedirs(output_images_folder, exist_ok=True)
os.makedirs(output_annotations_folder, exist_ok=True)

# Get annotation filenames (without extension)
annotation_files = {os.path.splitext(f)[0] for f in os.listdir(annotations_folder) if f.endswith(".txt")}

# Iterate through images and copy those with annotations
for image_file in os.listdir(images_folder):
    image_name, ext = os.path.splitext(image_file)
    if image_name in annotation_files:  # Check if corresponding annotation exists
        # Copy image
        shutil.copy(os.path.join(images_folder, image_file), os.path.join(output_images_folder, image_file))
        # Copy annotation
        shutil.copy(os.path.join(annotations_folder, image_name + ".txt"), os.path.join(output_annotations_folder, image_name + ".txt"))

print("Images and annotations copied successfully!")