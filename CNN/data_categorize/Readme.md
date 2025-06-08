# Waste Image Categorizer & Renamer

## Overview
This script is designed to categorize and rename waste images into specific classes and organize them into training and testing datasets. It also generates a CSV file containing the filenames and their corresponding labels for further use in machine learning models.

---

## Features
- Categorizes images into 12 predefined classes.
- Renames images with a class-specific prefix and a unique number.
- Moves images to the appropriate `train` or `test` folder.
- Updates and saves class counts to a file (`class_counts.txt`).
- Generates a CSV file (`train_labels.csv` or `test_labels.csv`) with filenames and labels.

---

## Predefined Classes
The script supports the following 12 categories:
1. Non-Recyclable
2. Plastic (10¢ rebate)
3. Plastic (25¢ rebate)
4. Plastic (No rebate)
5. Glass (10¢ rebate)
6. Glass (25¢ rebate)
7. Glass (No rebate)
8. Glass (High-value, $1 rebate)
9. Metal (10¢ rebate)
10. Metal (25¢ rebate)
11. Metal (No rebate)
12. Paper/Cardboard

---

## Usage
1. Place raw images in the `data/raw_image` folder.
2. Run the script:
   ```bash
   python categorize_and_rename.py

3. Follow the prompts:
    Specify whether the images are for the train or test set.
    Select the class number (0-11) for the images.
    
4. The script will:
    Rename and move the images to the appropriate folder (train/images or test/images).
    Update the class counts in class_counts.txt.
    Append the filenames and labels to the corresponding CSV file (train_labels.csv or test_labels.csv).