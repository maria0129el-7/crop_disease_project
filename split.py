import os
import shutil
import random

# -----------------------------
# Original Dataset Path
# -----------------------------
dataset_path = r"C:\Users\hp\Downloads\archive (4)\PlantVillage"

# -----------------------------
# Output Folder
# -----------------------------
output_path = "data"

# -----------------------------
# Class Names
# -----------------------------
classes = {
    "Potato___Early_blight": "early_blight",
    "Potato___Late_blight": "late_blight",
    "Potato___healthy": "healthy"
}

# -----------------------------
# Create Folders
# -----------------------------
for split in ["train", "val"]:

    for class_name in classes.values():

        folder_path = os.path.join(
            output_path,
            split,
            class_name
        )

        os.makedirs(folder_path, exist_ok=True)

# -----------------------------
# 80% Train / 20% Validation
# -----------------------------
train_ratio = 0.8

# -----------------------------
# Split Dataset
# -----------------------------
for original_class, new_class in classes.items():

    source_folder = os.path.join(
        dataset_path,
        original_class
    )

    images = os.listdir(source_folder)

    random.shuffle(images)

    split_index = int(len(images) * train_ratio)

    train_images = images[:split_index]

    val_images = images[split_index:]

    # -------- TRAIN --------
    for image in train_images:

        source_path = os.path.join(
            source_folder,
            image
        )

        destination_path = os.path.join(
            output_path,
            "train",
            new_class,
            image
        )

        shutil.copy(source_path, destination_path)

    # -------- VALIDATION --------
    for image in val_images:

        source_path = os.path.join(
            source_folder,
            image
        )

        destination_path = os.path.join(
            output_path,
            "val",
            new_class,
            image
        )

        shutil.copy(source_path, destination_path)

print("Dataset split completed successfully!")