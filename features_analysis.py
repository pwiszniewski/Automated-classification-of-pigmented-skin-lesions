# Load the Pandas libraries with alias 'pd' 
import pandas as pd 
import cv2
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import preprocessing

base_dir = 'images'

def normalize(df, columns):
    # num_cols = df.columns[df.dtypes.apply(lambda c: np.issubdtype(c, np.number))]
    # df[num_cols] = preprocessing.StandardScaler().fit_transform(df[num_cols])
    for feature_name in columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        df[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)

df_org = pd.read_csv("data/HAM10000_metadata_params_21012020.csv", index_col=0) 
#df_left = df.drop(df.columns[9:])

#print(df.iloc[:, 8:].head())
df = df_org.copy()
features = [*df.columns[7:]]
# features1 = ['hu'+str(i) for i in range(7)]
normalize(df, features)
fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(1, 1, 1)
df.boxplot(by='dx', 
            column=features, 
            grid=False,
           ax=ax)

plt.figure(figsize=(20,10))
df = df[['dx', *features]]
df =  pd.get_dummies(df, prefix_sep='_', columns=['dx'])
df_corr = df.corr()
sns.heatmap(df_corr.abs()[-7:].T, yticklabels=True, annot=True)
# plt.grid()
plt.tight_layout()



#for feat in features:
#    df_feat = df[['dx', feat]].boxplot(by='dx')
        
#df.head()
#corr = df.corr()
##corr.style.background_gradient(cmap='coolwarm')
##df.boxplot()
#categorical_feature_mask = df.dtypes==object
#categorical_cols = df.columns[categorical_feature_mask].tolist()
#
## import labelencoder
#from sklearn.preprocessing import LabelEncoder# instantiate labelencoder object
#le = LabelEncoder()

# apply le on categorical feature columns
#df[categorical_cols] = df[categorical_cols].apply(lambda col: le.fit_transform(col))

#for i in range(5):
#    fig = plt.figure()
#    df_dx = df[df.dx == i+1].iloc[:, 8:]
#    df_dx[].boxplot()
#    plt.show()
