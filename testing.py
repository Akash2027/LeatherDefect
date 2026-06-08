from ultralytics import YOLO
import cv2
import torch
import matplotlib.pyplot as plt

# Load the trained YOLOv11 model
model = YOLO("modelv1.pt")  # Ensure the correct path to your model

# Test image path
image_path = r"C:\Users\AdminUser\Desktop\WINTER SEM 24-25\LeatherDatasetPreprocessing\YOLODATASET\images\test\Folding marks 01 (492).jpg"


# Run inference
results = model(image_path)  # Run YOLO prediction

# Load image
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for proper visualization

# Iterate through detected objects
for result in results:
    for box in result.boxes:
        # Get bounding box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert to integers
        cls = int(box.cls[0].item())  # Get class index
        conf = box.conf[0].item()  # Get confidence score

        # Get class name from model
        class_name = model.names[cls]

        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)  # Blue box with thickness 3

        # Prepare label text
        label = f"{class_name} {conf:.2f}"

        # Get text size
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

        # Draw filled rectangle as background for text
        cv2.rectangle(img, (x1, y1 - text_height - 5), (x1 + text_width, y1), (255, 0, 0), -1)

        # Put class name text on the image
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

# Display image with bounding boxes
plt.figure(figsize=(8, 6))
plt.imshow(img)
plt.axis("off")
plt.show()
