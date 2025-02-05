Introduction
The Anole Classifier leverages machine learning techniques to accurately classify anole species based on images.
This project aims to aid the Stroud Lab's middle schooler citizen scientist program in identifying these reptiles.

Installation:

bash
cd anole-classifier
Install the required dependencies:

bash
pip install -r requirements.txt

Usage
To use the classifier, follow these steps:

Prepare your dataset and ensure it is organized as specified in the Dataset section.

Train the models:

bash
python train.py
python train_ensemble.py
python train_detection.py

Use the trained model to classify new images:

bash
python test.py 


Dataset
The dataset used for this project consists of labeled images of different anole species. The dataset should be organized into subfolders for each species, with images in each subfolder.
To download the dataset use the utility.py file. The file path must be changed to one of the csv files and the invoktion uncommented.

This dataset is generated using iNaturalist export: https://www.inaturalist.org/observations/export
Using queries: ?quality_grade=research&identifications=any&field%3Abanded=yes&taxon_ids%5B%5D=36488%2C36391%2C36455, ?quality_grade=research&identifications=any&field%3Abanded=yes&taxon_ids%5B%5D=11646%2C36514

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

Model
The classifier uses a convolutional neural network (CNN) architecture. The model is built using TensorFlow and Keras. The detailed architecture can be found in the train.py, train_ensemble.py, and train_detection.py files.

Results
The trained model achieves an accuracy of [X%] on the test set.
