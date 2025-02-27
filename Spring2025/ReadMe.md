Introduction
The Florida Anole Species Classification project aims to develop a robust machine learning pipeline for identifying five common Anolis species from photographs, primarily to support a community science initiative with middle school students in Miami. Building upon an extensive dataset of over 80,000 verified iNaturalist photographs, this project seeks to improve the current classification system, which, despite having access to substantial training data, currently achieves only 35% accuracy (compared to a random baseline of 20%). The development of this classification pipeline will serve as the foundation for a broader educational tool, whether implemented as a mobile application or web platform, that enables students to receive immediate probability-based species identification feedback before submitting their observations to iNaturalist, thereby enhancing the quality of citizen science data collection while engaging young students in herpetological research. 

The current method proposes a pipeline that consists of 1) Detecting lizard species using fine-tuned lizard detection model 2) Upscaling and enhancing cropped image 3) Classifying lizard subspecies using Classification model trained with cropped images.

Installation:

    bash
    cd Anole_classifier
    Install the required dependencies:

    bash
    conda create -f requirements_window.yml
    conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
    pip install "tensorflow<2.11" 

Dataset
Prepare your dataset and ensure it is organized as specified in the Dataset section.

To download the raw, labelled images from iNaturalist:

    The dataset used for this project consists of labeled images of different anole species. 
    The dataset should be organized into subfolders for each species, with images in each subfolder.
        To download the dataset: 
            1. Download the .csv file (Contains the taxonID, link, image metadata) 
            2. Use the utility.py file. 
                The file path must be changed to one of the csv files and the invoktion uncommented.

    Generate dataset .csv file using the following steps:
        - navigate to iNaturalist export: https://www.inaturalist.org/observations/export
        - use queries: ?quality_grade=research&identifications=any&field%3Abanded=yes&taxon_ids%5B%5D=36488%2C36391%2C36455, ?quality_grade=research&identifications=any&field%3Abanded=yes&taxon_ids%5B%5D=116461%2C36514

    Dataset information
        Data Count (Species Name, Taxon ID, Count)

            Knight Anole (Anolis Equestris), 36391, 2301
            Crested Anole (Anolis cristatellus), 36488, 6075
            Bark Anole (Anolis distichus), 36455, 2671
            Green Anole (Anolis carolinensis), 36514, 44242 (not fully downloaded)
            Brown Anole (Anolis sagrei), 116461, 44756 (not fully downloaded)

To upload to Roboflow:

$ roboflow login
$ roboflow import -w wen-han-chia-3l2oy -p anole_annotate /path/to/data


Example structure:

project/
    Raw/
        taxon_001.jpg
        taxon_002.jpg
        ...
    Raw-Test/
        taxon_001.jpg
        taxon_002.jpg
        ...

