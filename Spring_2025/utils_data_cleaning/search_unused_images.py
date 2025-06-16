import os
import bisect

def get_sorted_filenames(folder):
    """Return a sorted list of filenames in the given folder."""
    return sorted(os.listdir(folder))

def binary_search(sorted_list, target):
    """Binary search to check if target exists in sorted_list."""
    index = bisect.bisect_left(sorted_list, target)
    return index < len(sorted_list) and sorted_list[index] == target

def delete_duplicates(folder1, folder2, log_filename="deleted_duplicates_log.txt"):
    folder1_files = os.listdir(folder1)
    folder2_files = get_sorted_filenames(folder2)

    deleted_files = []

    for file in folder1_files:
        if binary_search(folder2_files, file):
            file_path = os.path.join(folder2, file)
            if os.path.exists(file_path):
                os.remove(file_path)
                deleted_files.append(file)
                print(f"Deleted duplicate from Folder 2: {file}")

    # Write log
    with open(log_filename, "w") as log_file:
        log_file.write(f"Total deleted: {len(deleted_files)}\n\n")
        for file in deleted_files:
            log_file.write(file + "\n")

    print(f"\nDone. {len(deleted_files)} file(s) deleted from '{folder2}'.")
    print(f"Log written to '{log_filename}'.")

if __name__ == "__main__":
    folder1_path = "../../Dataset/YOLO_training/base_data/original_test/brown_anole_2000/train/images"
    folder2_path = "../../Dataset/YOLO_training/base_data/original_test/BrownAnole"

    delete_duplicates(folder1_path, folder2_path)
