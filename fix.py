import os
import json

# Specify the folder containing the JSON files
annotations_folder = 'F:/LizardCV/bbox'
# Iterate through each JSON file in the specified folder
for filename in os.listdir(annotations_folder):
    if filename.endswith('.json'):
        json_path = os.path.join(annotations_folder, filename)
        
        # Load the JSON file
        with open(json_path, 'r') as f:
            annotation = json.load(f)

        # Update the imagePath by removing "..\\"
        if 'imagePath' in annotation:
            annotation['imagePath'] = annotation['imagePath'].replace('..\\', '')

        # Save the modified JSON back to the file
        with open(json_path, 'w') as f:
            json.dump(annotation, f, indent=4)

print("Updated imagePath in all JSON files.")