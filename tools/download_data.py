import os
import urllib
import sys
import zipfile
from pathlib import Path
import pandas as pd

def download_file(url, file_name):
	u = urllib.request.urlretrieve(url, file_name)

def convert_tab_to_csv(fname_tab, fname_csv):
  df = pd.read_csv(fname_tab,delimiter="\t")
  df.to_csv(fname_csv, encoding='utf-8', index=False)

def unzip_file(file_path, dest_path):
	with zipfile.ZipFile(file_path, 'r') as zip_ref:
	    zip_ref.extractall(dest_path)

def remove_file(file_path):
  if os.path.isfile(file_path):
	  os.remove(file_path)

images_path = 'images'
data_path = 'data'
Path(images_path).mkdir(parents=True, exist_ok=True)
Path(data_path).mkdir(parents=True, exist_ok=True)

files = {
	"HAM10000_metadata": ("https://dataverse.harvard.edu/api/access/datafile/3172582", data_path+"/HAM10000_metadata.tab"),
	"HAM10000_images_part_1": ("https://dataverse.harvard.edu/api/access/datafile/3172585", images_path+"/HAM10000_images_part_1.zip"),
	"HAM10000_images_part_2": ("https://dataverse.harvard.edu/api/access/datafile/3172584", images_path+"/HAM10000_images_part_2.zip"),
}

tab_path = files['HAM10000_metadata'][1]
csv_path = tab_path.rstrip('tab')+'csv'

for f in files:
	print('downloading', f)
	download_file(*files[f])

for f in ("HAM10000_images_part_1", "HAM10000_images_part_2"):
	file_path = files[f][1]
	unzip_file(file_path, images_path)
	remove_file(file_path)
	
convert_tab_to_csv(tab_path, csv_path)
