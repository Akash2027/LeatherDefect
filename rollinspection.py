import cv2
import os
import numpy as np
from ultralytics import YOLO

# Load trained YOLO model
model = YOLO("modelv1.pt")  # Ensure the correct path to your model

# Path to test images
test_images_path = r"C:\Users\AdminUser\Desktop\WINTER SEM 24-25\LeatherDatasetPreprocessing\YOLODATASET\images\test"
image_files = sorted([f for f in os.listdir(test_images_path) if f.endswith(('.jpg', '.png'))])

# Window settings
cv2.namedWindow("Leather Roll Inspection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Leather Roll Inspection", 1200, 800)

# Constants for bounding box size
BOX_EXPAND = 20  # Increase bounding box size for better visibility

# Background size
BG_WIDTH = 1200
BG_HEIGHT = 800

# Center of the background
CENTER_X = BG_WIDTH // 2
CENTER_Y = BG_HEIGHT // 2

# Pause flag
paused = False

def toggle_pause():
    global paused
    paused = not paused

# Start rolling effect
for img_name in image_files:
    img_path = os.path.join(test_images_path, img_name)

    # Load leather image
    leather_img = cv2.imread(img_path)
    leather_img = cv2.cvtColor(leather_img, cv2.COLOR_BGR2RGB)  # Convert for better visualization

    # Resize leather image to fit within 1200x800 while maintaining aspect ratio
    scale_factor = min(BG_WIDTH / leather_img.shape[1], BG_HEIGHT / leather_img.shape[0])
    new_width = int(leather_img.shape[1] * scale_factor)
    new_height = int(leather_img.shape[0] * scale_factor)
    leather_img = cv2.resize(leather_img, (new_width, new_height))

    # Create a black background
    black_bg = np.zeros((BG_HEIGHT, BG_WIDTH, 3), dtype=np.uint8)

    # Compute centering position
    start_x = (BG_WIDTH - new_width) // 2
    start_y = (BG_HEIGHT - new_height) // 2

    # Place the resized leather image at the center
    black_bg[start_y:start_y + leather_img.shape[0], start_x:start_x + leather_img.shape[1]] = leather_img

    # Run YOLO inference
    results = model(img_path)

    # Get detection results and center the bounding box
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get original bounding box coordinates

            # Compute original bounding box width and height
            bbox_width = x2 - x1
            bbox_height = y2 - y1

            # Expand bounding box for better visualization
            bbox_width += 15 * BOX_EXPAND
            bbox_height += 15 * BOX_EXPAND

            # Compute centered bounding box position
            centered_x1 = CENTER_X - bbox_width // 2
            centered_y1 = CENTER_Y - bbox_height // 2
            centered_x2 = centered_x1 + bbox_width
            centered_y2 = centered_y1 + bbox_height

            # Ensure bounding box stays within image bounds
            centered_x1 = max(0, centered_x1)
            centered_y1 = max(0, centered_y1)
            centered_x2 = min(BG_WIDTH, centered_x2)
            centered_y2 = min(BG_HEIGHT, centered_y2)

            # Get class name
            class_name = model.names[int(box.cls[0].item())]
            conf = box.conf[0].item()

            # Label text
            label = f"{class_name} ({conf:.2f})"

            # Draw bounding box in the center
            cv2.rectangle(black_bg, (centered_x1, centered_y1), (centered_x2, centered_y2), (0, 255, 0), 2)

            # Get text size
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

            # Draw filled rectangle for label background
            cv2.rectangle(
                black_bg,
                (centered_x1, centered_y1 - text_height - 5),
                (centered_x1 + text_width, centered_y1),
                (0, 255, 0),
                -1
            )

            # Put label text
            cv2.putText(black_bg, label, (centered_x1, centered_y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Display the final image on a black background
    cv2.imshow("Leather Roll Inspection", black_bg)

    while True:
        key = cv2.waitKey(200) & 0xFF
        if key == 32:  # Spacebar toggles pause
            toggle_pause()
        elif key == 27:  # Escape key exits
            cv2.destroyAllWindows()
            exit()
        if not paused:
            break

cv2.destroyAllWindows()
