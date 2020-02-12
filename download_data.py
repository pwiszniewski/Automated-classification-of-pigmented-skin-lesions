import urllib
import sys
import zipfile
from pathlib import Path

def download_file(url, file_name):
	u = urllib.request.urlretrieve(url, file_name)

images_path = 'images'
data_path = 'data'
Path(images_path).mkdir(parents=True, exist_ok=True)
Path(data_path).mkdir(parents=True, exist_ok=True)

files = {
	"HAM10000_metadata": ("https://dataverse.harvard.edu/api/access/datafile/3172582", data_path+"/HAM10000_metadata.tab"),
	"HAM10000_images_part_1": ("https://dataverse.harvard.edu/api/access/datafile/3172585", images_path+"/HAM10000_images_part_1.zip"),
	"HAM10000_images_part_2": ("https://dataverse.harvard.edu/api/access/datafile/3172584", images_path+"/HAM10000_images_part_2.zip"),
}

for f in files:
	download_file(*files[f])

for f in ("HAM10000_images_part_1", "HAM10000_images_part_2"):
	path = files[f][1]
	with zipfile.ZipFile(path, 'r') as zip_ref:
	    zip_ref.extractall(images_path)