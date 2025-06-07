import os
import shutil
import csv
from pathlib import Path

# Category mapping (class number to name for filename prefix)
CATEGORY_PREFIXES = [
    'nonrecyclable',
    'plastic_10',
    'plastic_25',
    'plastic_none',
    'glass_10',
    'glass_25',
    'glass_none',
    'glass_1dollar',
    'metal_10',
    'metal_25',
    'metal_none',
    'paper_cardboard'
]

SPLITS = ['train', 'test']

# Use absolute paths only
DATA_ROOT = Path('../data')
COUNTS_FILE = Path('class_counts.txt')
RAW_IMAGE_DIR = Path('raw_image')


def load_class_counts():
    counts = {i: 0 for i in range(12)}
    if COUNTS_FILE.exists():
        with open(COUNTS_FILE, 'r') as f:
            for line in f:
                parts = line.strip().split(':')
                if len(parts) == 2:
                    idx, val = int(parts[0]), int(parts[1])
                    counts[idx] = val
    return counts

def save_class_counts(counts):
    with open(COUNTS_FILE, 'w') as f:
        for idx in range(12):
            f.write(f'{idx}:{counts[idx]}\n')

def main():
    print('--- Waste Image Categorizer & Renamer ---')
    img_folder = RAW_IMAGE_DIR
    if not img_folder.exists() or not img_folder.is_dir():
        print(f'Image folder {img_folder} does not exist.')
        return

    split = input('Are these images for train or test set? (train/test): ').strip().lower()
    if split not in SPLITS:
        print('Invalid split. Must be "train" or "test".')
        return

    # Check that the output directory exists
    out_dir = DATA_ROOT / split / 'images'
    if not out_dir.exists() or not out_dir.is_dir():
        print(f'Output directory {out_dir} does not exist. Please create it first.')
        return
    csv_path = DATA_ROOT / split / f'{split}_labels.csv'

    print('Class options:')
    for idx, name in enumerate(CATEGORY_PREFIXES):
        print(f' {idx}: {name}')
    class_num = input('Enter the class number (0-11) for these images: ').strip()
    try:
        class_num = int(class_num)
        assert 0 <= class_num < 12
    except:
        print('Invalid class number.')
        return

    prefix = CATEGORY_PREFIXES[class_num]
    counts = load_class_counts()
    next_num = counts[class_num] + 1

    # Process images
    img_files = [f for f in os.listdir(img_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    if not img_files:
        print('No images found in the folder.')
        return

    csv_exists = csv_path.exists()
    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not csv_exists:
            writer.writerow(['filename', 'label'])
        for img_file in img_files:
            ext = os.path.splitext(img_file)[1].lower()
            new_name = f'{prefix}_{next_num:03d}{ext}'
            src = img_folder / img_file
            dst = out_dir / new_name
            try:
                shutil.move(str(src), dst)
                writer.writerow([new_name, class_num])
                print(f'Renamed and moved: {img_file} -> {new_name} (class {class_num})')
                next_num += 1
            except Exception as e:
                print(f'Error moving file {img_file}: {e}')
    
    # Update and save class counts
    counts[class_num] = next_num - 1
    save_class_counts(counts)
    print(f'Updated class {class_num} count: {counts[class_num]}')
    print('Done!')

if __name__ == '__main__':
    main()