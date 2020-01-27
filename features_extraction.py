# Load the Pandas libraries with alias 'pd' 
import pandas as pd 
from image_calculator_v2 import calc_parameters
import cv2
import os
from sklearn.utils import shuffle

base_dir = 'images'

data = pd.read_csv("data/HAM10000_metadata.csv") 
data = shuffle(data)
data.head()

#data = data[(data.localization == 'face') & (data.sex == 'female')]

series_list = []
it = 0
for _,d in data.iterrows():
    img_org = cv2.imread(os.path.join(base_dir, d.image_id+'.jpg'),1)
    histograms, params_dict = calc_parameters(img_org, show_params=False, 
                                              show_hist=False, show_img=True)
    for p in params_dict:
        d[p] = params_dict[p]
    series_list.append(d)
    it += 1
    if it % 100 == 0:
        print('->', it)
        # break
    # if it % 100 == 0:
    #     break
#    if cv2.waitKey(0) & 0xFF == ord('q'):
#        break
    
df = pd.DataFrame(series_list)
df['per_area_ratio'] = df['perimeter'] / df['max_area']
df.to_csv('HAM10000_metadata_params_15012020.csv')
