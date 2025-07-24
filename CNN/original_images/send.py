import os

base_dir = os.path.dirname(os.path.abspath(__file__))

for folder in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder)
    if os.path.isdir(folder_path) and not folder.startswith('.'):
        images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        images.sort()
        # Step 1: Rename all to temp names to avoid collision
        for img_name in images:
            src = os.path.join(folder_path, img_name)
            temp_name = f"__temp__{img_name}"
            dst = os.path.join(folder_path, temp_name)
            os.rename(src, dst)
        # Step 2: Rename temp files to final names
        temp_images = [f for f in os.listdir(folder_path) if f.startswith('__temp__')]
        temp_images.sort()
        for idx, temp_name in enumerate(temp_images, 1):
            ext = os.path.splitext(temp_name)[1]
            new_name = f"{folder}_{idx:03d}{ext}"
            src = os.path.join(folder_path, temp_name)
            dst = os.path.join(folder_path, new_name)
            os.rename(src, dst)
            print(f"Renamed: {src} -> {dst}")