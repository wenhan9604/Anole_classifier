import json
import os

# Define the folder containing the JSON files
folder_path = "F:/LizardCV/bbox"

# Loop through each file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".json"):
        file_path = os.path.join(folder_path, filename)
        
        # Load JSON data
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Process each shape to adjust points for top-left and bottom-right
        for shape in data.get("shapes", []):
            # Extract points
            x1, y1 = shape["points"][0]
            x2, y2 = shape["points"][1]
            
            # Calculate top-left and bottom-right
            top_left_x = min(x1, x2)
            top_left_y = min(y1, y2)
            bottom_right_x = max(x1, x2)
            bottom_right_y = max(y1, y2)
            
            # Update points to be top-left and bottom-right
            shape["points"] = [
                [top_left_x, top_left_y],    # Top-left
                [bottom_right_x, bottom_right_y]  # Bottom-right
            ]

        # Overwrite the existing file with updated JSON data
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)

        print(f"Bounding boxes updated in {file_path}")
