import os
import random
import sys
import re

def rename_images_with_size_suffix(img_root_dir):
    print("🔄 Renaming images by adding _10/_20 ...")

    for class_name in os.listdir(img_root_dir):
        class_dir = os.path.join(img_root_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        files.sort()

        for idx, original_name in enumerate(files, start=1):
            size = random.choice(["10", "20"])
            ext = os.path.splitext(original_name)[1]
            new_name = f"{class_name}_{size}_{idx:03d}{ext}"

            src_path = os.path.join(class_dir, original_name)
            dst_path = os.path.join(class_dir, new_name)

            os.rename(src_path, dst_path)

    print("✅ All image files renamed with _10/_20.")

def undo_image_rename(img_root_dir):
    print("🔄 Restoring original filenames ...")

    pattern = re.compile(r"^(?P<class>\w+)_\d{2}_(?P<index>\d{3})(\.\w+)$")

    for class_name in os.listdir(img_root_dir):
        class_dir = os.path.join(img_root_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        for filename in os.listdir(class_dir):
            match = pattern.match(filename)
            if not match:
                continue

            original_name = f"{match.group('class')}_{match.group('index')}{os.path.splitext(filename)[1]}"
            src_path = os.path.join(class_dir, filename)
            dst_path = os.path.join(class_dir, original_name)

            os.rename(src_path, dst_path)

    print("✅ All filenames restored to original format (class_index.jpg)")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("❌ Usage: python script.py [rename|undo]")
        sys.exit(1)

    mode = sys.argv[1].lower()
    root_dir = "original_images"

    if mode == "rename":
        rename_images_with_size_suffix(root_dir)
    elif mode == "undo":
        undo_image_rename(root_dir)
    else:
        print("❌ Invalid argument. Use 'rename' or 'undo'.")
