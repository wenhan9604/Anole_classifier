import urllib.request
import requests
import json
import pandas as pd
import numpy as np
import threading
import urllib
import os
import cv2

#configs
taxa = [116461, 36514, 36488, 36391, 36455]
folder_path = 'F:/LizardCV'
file_path = 'F:/LizardCV/36488&36391&36455.csv' #modify with file name

def download_images(filepath):
    df = pd.read_csv(filepath)
    for arr in np.array_split(df, 10):
        df = pd.DataFrame(arr)
        t = threading.Thread(target=download_images_part, args=(arr,))
        t.start()

def download_images_part(df):
    for index, row in df.iterrows():
        taxon_id = row['taxon_id']
        image_url = row['image_url']
        id = row['id']

        if not pd.isnull(image_url) and 'https://inaturalist-open-data.s3.amazonaws.com/photos' in image_url:
            save_path = f"{folder_path}/{taxon_id}/{taxon_id}_{id}.jpg"
            os.makedirs( os.path.dirname(save_path),exist_ok=True)
            if not os.path.isfile(save_path):
                urllib.request.urlretrieve(image_url, save_path)
        if index%1000 == 0:
            print("In Progress")

def resize():
    for subdir, dirs, images in os.walk(folder_path):
        for img in images:
            print(img)
            if img.endswith('.jpg'):
                im = cv2.imread(os.path.join(subdir,img))
                if im is not None and im.shape != (256,256):
                    im = cv2.resize(im,(256, 256))
                    cv2.imwrite(os.path.join(subdir,img),im)


resize()