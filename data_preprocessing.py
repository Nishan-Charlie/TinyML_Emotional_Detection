import os
import shutil
from sklearn.model_selection import train_test_split

# Define your paths
root_dir = "images"
train_path = os.path.join(root_dir, "train")
val_path = os.path.join(root_dir, "valid")

# 1. Create the validation directory if it doesn't exist
os.makedirs(val_path, exist_ok=True)

# 2. Iterate over each class directory in the train path
classes = [d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))]

print(f"Found classes: {classes}")

for class_name in classes:
    class_train_dir = os.path.join(train_path, class_name)
    class_val_dir = os.path.join(val_path, class_name)
    
    # Create the class folder inside valid directory
    os.makedirs(class_val_dir, exist_ok=True)
    
    # List all files in the current class directory
    files = os.listdir(class_train_dir)
    
    # Check if there are enough files to split
    if len(files) < 2:
        print(f"Skipping {class_name}: Not enough images to split.")
        continue

    # 3. Split the files (90% train, 10% valid)
    # Since we are looping per class, this is automatically a stratified split
    train_files, val_files = train_test_split(files, test_size=0.1, random_state=42)
    
    print(f"Moving {len(val_files)} images for class '{class_name}' to validation...")
    
    # 4. Move the validation files to the new directory
    for file in val_files:
        src = os.path.join(class_train_dir, file)
        dst = os.path.join(class_val_dir, file)
        shutil.move(src, dst)

print("---")
print("Splitting complete.")
print(f"Train data at: {train_path}")
print(f"Valid data at: {val_path}")