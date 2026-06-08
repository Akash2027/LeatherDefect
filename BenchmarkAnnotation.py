import os

# Define paths
image_folder = "./LeatherDefectBenchmarkDataset/pinhole"  # Update this with the actual path
annotation_folder = "./LeatherDefectBenchmarkDataset/annotates"  # Update this with the actual path

# Ensure annotation folder exists
os.makedirs(annotation_folder, exist_ok=True)

# Predefined bounding box values for FoldMark (as given)
annotation_content = "4 0.500000 0.500000 0.938326 0.938326\n"

# Loop through all images in the image folder
for filename in os.listdir(image_folder):
    if filename.endswith((".jpg", ".png", ".jpeg")):  # Consider only image files
        # Remove the file extension to create annotation filename
        name_without_ext = os.path.splitext(filename)[0]

        # Path for the annotation file
        annotation_path = os.path.join(annotation_folder, name_without_ext + ".txt")

        # Write the annotation content
        with open(annotation_path, "w") as file:
            file.write(annotation_content)

print(f"Annotations generated and saved in: {annotation_folder}")
