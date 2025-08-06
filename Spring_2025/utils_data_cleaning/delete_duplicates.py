import os
import bisect

def get_sorted_filenames(folder):
    """Return a sorted list of filenames in the given folder."""
    return sorted(os.listdir(folder))

def binary_search(sorted_list, target):
    """Binary search to check if target exists in sorted_list."""
    index = bisect.bisect_left(sorted_list, target)
    return index < len(sorted_list) and sorted_list[index] == target

def delete_duplicates(reference_folder_path, edited_folder_path, file_extension="jpg", log_filename="deleted_duplicates_log.txt"):
    '''
    Summary: Remove files in edited folder based on file names found in reference folder. 
        - Performs binary search for item
    '''

    if not os.path.isdir(reference_folder_path) or not os.path.isdir(edited_folder_path):
        print(f"Input file path does not exist! Exiting function")
        exit()

    ref_folder = os.listdir(reference_folder_path)
    edited_folder = get_sorted_filenames(edited_folder_path)

    deleted_files = []

    for file in ref_folder:

        file_name, extension = os.path.splitext(file)
        file = f"{file_name}.{file_extension}"

        if binary_search(edited_folder, file):
            file_path = os.path.join(edited_folder_path, file)
            os.remove(file_path)
            deleted_files.append(file)
            print(f"Deleted duplicate from ref folder: {file}")

    # Write log
    with open(log_filename, "w") as log_file:
        log_file.write(f"Total deleted: {len(deleted_files)}\n\n")
        for file in deleted_files:
            log_file.write(file + "\n")

    print(f"\nDone. {len(deleted_files)} file(s) deleted from '{edited_folder_path}'.")
    print(f"\nLog written to '{log_filename}'.")

if __name__ == "__main__":
    ref_folder = "../Dataset/YOLO_training/dataset_v3/original_cleaned/florida_10000_cleaned_verified/bark_anole_2000_verified/wenhan_bark_anole/test_labels - Copy"
    edited_folder_path = "../Dataset/YOLO_training/dataset_v3/original_cleaned/florida_10000_cleaned_verified/bark_anole_2000_verified/wenhan_bark_anole/test_labels"
    # edited_folder_path = "../Dataset/YOLO_training/dataset_v3/original_cleaned/sent_for_cleaning/raw_dataset_unused/BarkAnole_unused/"

    delete_duplicates(ref_folder, edited_folder_path, file_extension="txt")
