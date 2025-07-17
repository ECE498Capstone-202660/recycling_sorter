import os
import random

def rename_images_with_size_suffix(img_root_dir):

    print("renaming...")
    
    for class_name in os.listdir(img_root_dir):
        class_dir = os.path.join(img_root_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        files.sort()  # optional for stable renaming

        for idx, original_name in enumerate(files, start=1):
            size = random.choice(["10", "20"])  # Random size assignment
            ext = os.path.splitext(original_name)[1]
            new_name = f"{class_name}_{size}_{idx:03d}{ext}"

            src_path = os.path.join(class_dir, original_name)
            dst_path = os.path.join(class_dir, new_name)

            os.rename(src_path, dst_path)

    print("All image files renamed with _10/_20.")

if __name__ == "__main__":
    rename_images_with_size_suffix("original_images")