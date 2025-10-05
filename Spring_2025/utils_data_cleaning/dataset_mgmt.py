import os
import shutil
from pathlib import Path

def move_image_and_text_files(src_folder, dest_folder, n=1600):
    """
    Summary : 
        - Will move images and text files under source folder's sub directory (images and labels) into destination folder's sub directory (images and labels)
        - n = last number of files to be copied over. If < 0, will copy all files in the folder
    """
    # Ensure source and destination folders exist
    src_img_path = Path(str(src_folder + "/images"))
    src_txt_path = Path(str(src_folder + "/labels"))
    dest_img_path = Path(str(dest_folder + "/images"))
    dest_txt_path = Path(str(dest_folder + "/labels"))
    
    if not src_img_path.is_dir():
        print(f"Source folder '{src_folder}' does not exist.")
        return
    
    # Create destination folder if it doesn't exist
    dest_img_path.mkdir(parents=True, exist_ok=True)
    dest_txt_path.mkdir(parents=True, exist_ok=True)  
    
    # Get all files sorted by modification time (most recent first)
    images = sorted(src_img_path.glob('*'), key=lambda img_file: img_file.name)
    
    # Filter to ensure we only copy files, not directories
    images = [img_file for img_file in images if img_file.is_file()]

    if(n < 0): #Copy all image

        for image in images:
            img_name = image.stem

            #Move image
            shutil.move(str(image), str(dest_img_path / image.name))
            # print(f"Copied image: {image} -> {dest_img_path}")

            #Move corresponding label text file
            txt_file_name = img_name + ".txt"
            corresponding_txt_file_path = str(src_txt_path/txt_file_name)
            shutil.move(str(corresponding_txt_file_path), str(dest_txt_path/txt_file_name))
            # print(f"Copied txt file: {corresponding_txt_file_path} -> {dest_txt_path}")

    else: # Copy the last 'n' files

        for image in images[-n:]:
            img_name = image.stem

            #Move image
            shutil.move(str(image), str(dest_img_path / image.name))
            # print(f"Copied image: {image} -> {dest_img_path}")

            #Move corresponding label text file
            txt_file_name = img_name + ".txt"
            corresponding_txt_file_path = str(src_txt_path/txt_file_name)
            shutil.move(str(corresponding_txt_file_path), str(dest_txt_path/txt_file_name))
            # print(f"Copied txt file: {corresponding_txt_file_path} -> {dest_txt_path}")


    print(f"Successfully moved {len(images[-n:])} files from {src_folder} into {dest_folder}")


def copy_image_and_text_files(src_folder, dest_folder, n=1600):
    """
    Summary : 
        - Will copy images and text files under source folder's sub directory (images and labels) into destination folder's sub directory (images and labels)
        - n = last number of files to be copied over. If < 0, will copy all files in the folder
    """
    # Ensure source and destination folders exist
    src_img_path = Path(str(src_folder + "/images"))
    src_txt_path = Path(str(src_folder + "/labels"))
    dest_img_path = Path(str(dest_folder + "/images"))
    dest_txt_path = Path(str(dest_folder + "/labels"))
    
    if not src_img_path.is_dir():
        print(f"Source folder '{src_folder}' does not exist.")
        return
    
    # Create destination folder if it doesn't exist
    dest_img_path.mkdir(parents=True, exist_ok=True)
    dest_txt_path.mkdir(parents=True, exist_ok=True)  
    
    # Get all files sorted by modification time (most recent first)
    images = sorted(src_img_path.glob('*'), key=lambda img_file: img_file.name)
    
    # Filter to ensure we only copy files, not directories
    images = [img_file for img_file in images if img_file.is_file()]
    
    # Copy the last 'n' files

    if(n <0):

        for image in images:
            img_name = image.stem

            #copy image
            shutil.copy(str(image), str(dest_img_path / image.name))
            # print(f"Copied image: {image} -> {dest_img_path}")

            #copy corresponding label text file
            txt_file_name = img_name + ".txt"
            corresponding_txt_file_path = str(src_txt_path/txt_file_name)
            shutil.copy(str(corresponding_txt_file_path), str(dest_txt_path/txt_file_name))
            # print(f"Copied txt file: {corresponding_txt_file_path} -> {dest_txt_path}")

    else:

        for image in images[-n:]:
            img_name = image.stem

            #copy image
            shutil.copy(str(image), str(dest_img_path / image.name))
            # print(f"Copied image: {image} -> {dest_img_path}")

            #copy corresponding label text file
            txt_file_name = img_name + ".txt"
            corresponding_txt_file_path = str(src_txt_path/txt_file_name)
            shutil.copy(str(corresponding_txt_file_path), str(dest_txt_path/txt_file_name))
            # print(f"Copied txt file: {corresponding_txt_file_path} -> {dest_txt_path}")


    print(f"Successfully copied {len(images[-n:])} files into {dest_folder}")

# For moving into another file 

def move_image_files(src_folder, dest_folder, n=200):
    # Ensure source and destination folders exist
    src_img_path = Path(str(src_folder))
    dest_img_path = Path(str(dest_folder))
    
    if not src_img_path.is_dir():
        print(f"Source folder '{src_folder}' does not exist.")
        return
    
    # Create destination folder if it doesn't exist
    dest_img_path.mkdir(parents=True, exist_ok=True)
    
    # Get all files sorted by modification time (most recent first)
    images = sorted(src_img_path.glob('*'), key=lambda img_file: img_file.name)
    
    # Filter to ensure we only copy files, not directories
    images = [img_file for img_file in images if img_file.is_file()]
    
    # move the last 'n' files
    for image in images[-n:]:
        #Move image
        shutil.move(str(image), str(dest_img_path / image.name))
        # print(f"Copied image: {image} -> {dest_img_path}")

    print(f"Successfully moved {len(images[-n:])} files into {dest_folder}")

def move_image_files_and_test_using_ref(src_folder, dest_folder, ref_folder):
    """
    Will utilize the file_names from ref_folder as the reference to denote the source files to copy over to dest_folder
    """
    # Ensure source and destination folders exist
    src_img_folder_path = Path(str(src_folder + "/images"))
    src_labels_folder_path = Path(str(src_folder + "/labels"))
    dest_img_folder_path = Path(str(dest_folder + "/images"))
    dest_labels_folder_path = Path(str(dest_folder + "/labels"))

    ref_img_path = Path(str(ref_folder))
    
    if not src_img_folder_path.is_dir():
        print(f"Source folder '{src_folder}' does not exist.")
        return
    
    if not ref_img_path.is_dir():
        print(f"Reference folder '{ref_img_path}' does not exist.")
        return
    
    # Create destination folder if it doesn't exist
    dest_img_folder_path.mkdir(parents=True, exist_ok=True)
    dest_labels_folder_path.mkdir(parents=True, exist_ok=True)
    
    # Get all files sorted by modification time (most recent first)
    ref_images = sorted(ref_img_path.glob('*'), key=lambda img_file: img_file.name)
    
    # Filter to ensure we only copy files, not directories
    ref_images = [img_file for img_file in ref_images if img_file.is_file()]
    
    for ref_image in ref_images:
        ref_img_name = ref_image.stem

        print(f"ref file name: {ref_img_name}")

        if(ref_img_name.endswith('1')):
            print(f"skipped file name: {ref_img_name}")
            continue
            
        src_img_name = '_'.join(ref_img_name.split('_')[:-2]) 
        print(f"src file name: {src_img_name}")

        src_img_name_suffix = src_img_name + ".jpg"
        src_img_path = src_img_folder_path / src_img_name_suffix 
        dest_img_path = dest_img_folder_path / src_img_name_suffix

        if not src_img_path.exists():
            print(f"Warning: Source image not found: {src_img_path}")
            continue

        #Move image
        shutil.move(src_img_path, dest_img_path)
        print(f"Copied image: {src_img_path} -> {dest_img_path}")

        #Move corresponding label text file
        label_file_name = src_img_name + ".txt"
        src_label_file_path = str(src_labels_folder_path/label_file_name)
        dest_label_file_path = str(dest_labels_folder_path/label_file_name)

        shutil.move(str(src_label_file_path), dest_label_file_path)
        print(f"Copied txt file: {src_label_file_path} -> {dest_labels_folder_path}")


    print(f"Successfully copied {len(ref_images)} files into {dest_folder}")

# For copying and splitting 1 source folder into destination folder's sub folder 
# source_parent_folder = "../Dataset/YOLO_training/dataset_v4/original_v4/bark_anole"
# destination_parent_folder = "../Dataset/YOLO_training/dataset_v4/lizard_10000_v4/"

# sub_directory = ['test', 'valid', 'train']
# for folder_name in sub_directory:

#     source_folder_path = f"{source_parent_folder}/{folder_name}"
#     destination_folder_path = f"{destination_parent_folder}/{folder_name}"

#     if (folder_name == 'train'):
#         count = 1600
#     else:
#         count = 200

#     copy_image_and_text_files(source_folder_path, destination_folder_path, count)

source_parent_folder = "../Dataset/YOLO_training/dataset_v4/original_v4/unsplit_copy"
destination_parent_folder = "../Dataset/YOLO_training/dataset_v4/lizard_10000_v4/"

source_sub_directory = ['bark_anole', 'brown_anole', 'crested_anole', 'green_anole', 'knight_anole']
destination_sub_directory = ['test', 'valid', 'train']

for species in source_sub_directory:
    for split in destination_sub_directory:

        source_folder_path = f"{source_parent_folder}/{species}"
        destination_folder_path = f"{destination_parent_folder}/{split}"

        if (split == 'train'):
            count = 1600
        else:
            count = 200

        move_image_and_text_files(source_folder_path, destination_folder_path, count)