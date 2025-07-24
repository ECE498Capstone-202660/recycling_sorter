import os
import shutil
import random

def split_data(original_images_dir, data_dir, train_ratio=0.7, val_ratio=0.2):
    """Split images into train, val, and test sets."""
    class_mapping = {
        "cardboard": 0,
        "glass": 1,
        "metal": 2,
        "paper": 3,
        "plastic": 4,
        "trash": 5
    }

    # Create directories for train, val, and test splits
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(data_dir, split)
        img_dir = os.path.join(split_dir, "images")
        os.makedirs(img_dir, exist_ok=True)

    # Process each class
    for class_name, class_id in class_mapping.items():
        class_dir = os.path.join(original_images_dir, class_name)
        if os.path.exists(class_dir):
            images = [img for img in os.listdir(class_dir) if img.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
            random.shuffle(images)

            # Split images into train, val, and test sets
            train_count = int(len(images) * train_ratio)
            val_count = int(len(images) * val_ratio)

            train_images = images[:train_count]
            val_images = images[train_count:train_count + val_count]
            test_images = images[train_count + val_count:]

            # Copy images to respective folders
            for split, split_images in zip(["train", "val", "test"], [train_images, val_images, test_images]):
                split_dir = os.path.join(data_dir, split)
                img_dir = os.path.join(split_dir, "images")
                for img_file in split_images:
                    src = os.path.join(class_dir, img_file)
                    dst = os.path.join(img_dir, img_file)
                    shutil.copy(src, dst)

if __name__ == "__main__":
    original_images_dir = "C:/Users/leozz/OneDrive/Desktop/ECE498/recycling_sorter/CNN/original_images"
    data_dir = "C:/Users/leozz/OneDrive/Desktop/ECE498/recycling_sorter/CNN/data"
    split_data(original_images_dir, data_dir)
    print("Data split completed!")