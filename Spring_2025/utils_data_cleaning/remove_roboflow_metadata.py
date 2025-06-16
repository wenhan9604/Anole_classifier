import os
from collections import defaultdict

def clean_jpg_filenames(directory):
    duplicates = defaultdict(int)

    for filename in os.listdir(directory):
        if '_jpg.rf.' in filename:
            base_name = filename.split('_jpg.rf.')[0]
            new_name = base_name + '.jpg'
            src = os.path.join(directory, filename)
            dst = os.path.join(directory, new_name)

            if os.path.exists(dst):
                duplicates[new_name] += 1
                os.remove(src)
                print(f"Deleted (duplicate): {filename}")
            else:
                os.rename(src, dst)
                print(f"Renamed: {filename} → {new_name}")

    total_duplicates = sum(duplicates.values())

    # Write duplicates to a log file
    with open("duplicates_log.txt", "w") as log_file:
        log_file.write(f"Total duplicates: {total_duplicates}\n\n")
        for name, count in duplicates.items():
            log_file.write(f"{name}: {count} duplicate(s)\n")

    print(f"Total duplicates: {total_duplicates}\n")
    print(f"Duplicate info saved to 'duplicates_log.txt'.")

import os
from collections import defaultdict

def clean_txt_filenames(directory):
    duplicates = defaultdict(int)

    for filename in os.listdir(directory):
        if '_jpg.rf.' in filename:
            base_name = filename.split('_jpg.rf.')[0]
            new_name = base_name + '.txt'
            src = os.path.join(directory, filename)
            dst = os.path.join(directory, new_name)

            if os.path.exists(dst):
                duplicates[new_name] += 1
                os.remove(src)
                print(f"Deleted (duplicate): {filename}")
            else:
                os.rename(src, dst)
                print(f"Renamed: {filename} → {new_name}")

    total_duplicates = sum(duplicates.values())

    # Write duplicates to a log file
    with open("txt_duplicates_log.txt", "w") as log_file:
        log_file.write(f"Total duplicates: {total_duplicates}\n\n")
        for name, count in duplicates.items():
            log_file.write(f"{name}: {count} duplicate(s)\n")

    print(f"Total duplicates: {total_duplicates}")
    print("Duplicate info saved to 'txt_duplicates_log.txt'.")

if __name__ == "__main__":
    # folder_path = "../../Dataset/YOLO_training/base_data/original_test/greenanole_2000/train/labels"  # Update path if needed
    # clean_txt_filenames(folder_path)

    folder_path = "../../Dataset/YOLO_training/base_data/original_test/Florida_10000_RemovedDuplicates/knightanole_2000/train/labels"  # Change this if needed
    clean_txt_filenames(folder_path)
