from pathlib import Path

def edit_label(root_dir, target_char):
    """
    Arg:
        root_dir: Will edit all text files found under root directory 
        target_char: Will replace the first char of each line with target_char
    """

    root_dir_path = Path(root_dir)

    for txt_file in root_dir_path.glob("**/*.txt"):  # "**/*.txt" finds all .txt files recursively
        print(f"Processing {txt_file}")
        
        with open(txt_file, 'r+') as file:
            lines = file.readlines()
            file.seek(0)
            for line in lines:
                if line.strip():
                    file.write(target_char + line[1:])
                else:
                    file.write(line)
            file.truncate()


root_dir = "C:/Projects/OMSCS/Lizard_Classification/Anole_classifier/Dataset/YOLO_training/dataset_v4/florida_five_anole_10000_v4/per_species_split/knight_anole"

target_char = "4"

edit_label(root_dir=root_dir, target_char=target_char)