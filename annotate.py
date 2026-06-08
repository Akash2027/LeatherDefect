from ultralytics import YOLO
import cv2
import os
from glob import glob

# Load pre-trained YOLO model (Replace with your model)
model = YOLO("base_model.pt")

# Input and output directories
input_images_folder = "testdataset"  # Folder with images to annotate
output_annotations_folder = "Auto_Annotations"  # Folder to save YOLO format annotations

# Create output directory if not exists
os.makedirs(output_annotations_folder, exist_ok=True)

# Get all image paths
image_paths = glob(os.path.join(input_images_folder, "*.jpg")) + glob(os.path.join(input_images_folder, "*.png"))

# Run YOLO prediction on each image
for img_path in image_paths:
    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠ Skipping {img_path} (Could not load)")
        continue

    # Run inference
    results = model.predict(img_path, conf=0.3)  # Adjust confidence threshold if needed

    # Get filename without extension
    base_filename = os.path.splitext(os.path.basename(img_path))[0]
    annotation_path = os.path.join(output_annotations_folder, base_filename + ".txt")

    # Open file to save annotations
    with open(annotation_path, "w") as f:
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0].item())  # Class ID
                x_center, y_center, width, height = box.xywhn[0].tolist()  # YOLO format (normalized)

                # Write to YOLO format annotation file
                f.write(f"{cls} {x_center} {y_center} {width} {height}\n")

    print(f"✅ Annotated {img_path} -> {annotation_path}")

print("🎯 Auto-annotation completed successfully!")
