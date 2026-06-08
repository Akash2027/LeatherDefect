from PIL import Image
import os

input_dir = "LEATHERDATASETv1"  # Change this to your actual folder path

for filename in os.listdir(input_dir):
    if filename.lower().endswith(".jpeg"):
        old_path = os.path.join(input_dir, filename)
        new_filename = filename.replace(".jpeg", ".jpg")
        new_path = os.path.join(input_dir, new_filename)

        try:
            # Open the image and re-save it as a proper .jpg
            img = Image.open(old_path)
            img = img.convert("RGB")  # Ensure proper format
            img.save(new_path, "JPEG")  # Save as proper .jpg
            os.remove(old_path)  # Remove old corrupted file
            print(f"Converted: {filename} → {new_filename}")
        except Exception as e:
            print(f"⚠ Error converting {filename}: {e}")

print("✅ All .jpeg images have been successfully converted to .jpg!")
