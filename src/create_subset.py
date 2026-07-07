import os     
import shutil #for copying files from one place to another
import random 

SOURCE_DIR = r"C:\Users\labha\Downloads\gesture_dataset_china\gesture_datast"
DEST_DIR = r"C:\Users\labha\Downloads\gesture_dataset_china\gesture_subset"
SUBSET_RATIO = 0.2  # 20 % of each folder

os.makedirs(DEST_DIR, exist_ok=True) #creates destination folder if exists, otherwise ignore

gesture_folders = [
    f for f in os.listdir(SOURCE_DIR)
    if os.path.isdir(os.path.join(SOURCE_DIR, f)) #keep only folders, not files
]

print(f"Found {len(gesture_folders)} gesture folders:")
print(gesture_folders)
print("-" * 50)

#Initialize counters
total_copied = 0
total_size_bytes = 0

#Loop through each gesture folder
for gesture in sorted(gesture_folders):
    source_path = os.path.join(SOURCE_DIR, gesture)
    dest_path = os.path.join(DEST_DIR, gesture)
    os.makedirs(dest_path, exist_ok=True) #creates destination folder if exists, otherwise ignore

    # Get all video files in this gesture folder
    all_files = [
        f for f in os.listdir(source_path)
        if f.lower().endswith(('.avi'))
    ]
    # Calculate how many to copy (at least 1)
    num_to_copy = max(1, int(len(all_files) * SUBSET_RATIO))

    # Randomly select files
    random.seed(42)  # Fixed seed for reproducibility
    selected_files = random.sample(all_files, num_to_copy)

    # Copy selected files
    for file_name in selected_files:
        src_file = os.path.join(source_path, file_name)
        dst_file = os.path.join(dest_path, file_name)
        shutil.copy2(src_file, dst_file)
        total_size_bytes += os.path.getsize(src_file)
    total_copied += num_to_copy
    print(f"  {gesture:12s} | {len(all_files):4d} total -> {num_to_copy:4d} copied")

# Step 4: Print summary
print("-" * 50)
print(f"Total files copied: {total_copied}")
print(f"Total subset size:  {total_size_bytes / (1024**3):.2f} GB")
print(f"Saved to: {DEST_DIR}")