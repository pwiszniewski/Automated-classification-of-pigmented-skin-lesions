import pandas as pd 
# import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Ewaluacja
from sklearn import metrics
from sklearn.metrics import confusion_matrix, precision_score
from sklearn.metrics import recall_score, f1_score, accuracy_score
from sklearn.model_selection import cross_val_score

from sklearn.svm import SVC
from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC


# machine learning
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# Ewaluacja
from sklearn.metrics import confusion_matrix, precision_score
from sklearn.metrics import recall_score, f1_score, accuracy_score

# machine learning
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


def prepare_dataframes(part=False):
    df_org = pd.read_csv("data/HAM10000_metadata_params_ext.csv") 
    from sklearn.utils import shuffle
    if part:
        gb = df_org.groupby('dx')    
        df_gb = [gb.get_group(x)[:100] for x in gb.groups]
        df = pd.concat(df_gb)
    else:
        df = df_org
    df = shuffle(df)
    #print(df.groupby('dx').count())
    #df = df[:100]
    df = df.iloc[:, 3:]
    df = df.drop(df.columns[1:5], axis=1)
#    df = 
    return df_org, df

def prepare_X_y(df, auto=True, norm=True):
    y = df['dx']
    if auto:
        X = df.iloc[:, 1:]
#        X['sex'] = pd.get_dummies(X['sex'], prefix_sep='_', drop_first=True)
#        X = pd.get_dummies(X, prefix_sep='_', drop_first=False)
        clf = LinearSVC(C=0.01, penalty="l1", class_weight='balanced', dual=False).fit(X, y)
        model = SelectFromModel(clf, prefit=True)
        X_arr = model.transform(X)
        feature_idx = model.get_support()
        feature_names = X.columns[feature_idx]
        X = pd.DataFrame(X_arr, columns=feature_names)
    else:
        X = df.iloc[:, 1:5] #X = df.iloc[:, 1:9]
#        X['sex'] = pd.get_dummies(X['sex'], prefix_sep='_', drop_first=True)
#        X = pd.get_dummies(X, prefix_sep='_', drop_first=False)
    def normalize(df, columns):
        for feature_name in columns:
            max_value = df[feature_name].max()
            min_value = df[feature_name].min()
            df[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    if norm:
        num_cols = X.columns[X.dtypes.apply(lambda c: np.issubdtype(c, np.number))]
        X[num_cols] = preprocessing.StandardScaler().fit_transform(X[num_cols])
        # normalize(X, num_cols)
    return X, y
    
    
def classify(X_train, y_train, X_test, cross_val=True):
    clf = SVC(kernel='rbf', gamma='scale', class_weight='balanced')
    if cross_val:
        scores = cross_val_score(clf, X, y, cv=5)
        print('--> ', scores)
        print("Cross validation accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    return pred, clf.classes_

def classify_neuron(X_train, y_train, X_test):
  y_train = pd.get_dummies(y_train, prefix_sep='_', drop_first=False)
  model = Sequential()
  model.add(Dense(2048, input_shape=(X_train.shape[1],), activation='relu'))
  model.add(Dropout(0.25))
  model.add(Dense(256, activation='relu'))
  model.add(Dropout(0.25))
  model.add(Dense(64, activation='relu'))
  model.add(Dropout(0.25))
  model.add(Dense(16, activation='relu'))
  model.add(Dropout(0.25))
  # model.add(Dense(4, activation='relu'))
  # model.add(Dropout(0.25))
  model.add(Dense(7, activation='sigmoid'))
  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
  history = model.fit(X_train, y_train, batch_size=64, epochs=2, shuffle=True)
  model.fit(X_train, y_train)
  sns.lineplot(range(len(history.history['loss'])), history.history['loss'])
  # sns.lineplot(range(len(history.history['accuracy'])), history.history['accuracy'])
  pred = model.predict(X_test) > 0.5
  classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
  pred = [classes[np.argmax(dat)] for dat in pred]
  return pred, classes

def calculate_metrics(pred):
    cm = confusion_matrix(y_test, pred)
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] # normalise
    acc = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred, average=None)
    recall = recall_score(y_test, pred, average=None)
    f_score = f1_score(y_test, pred, average=None)
    return acc, precision, recall, f_score, cm, cmn

def calculate_metrics_neuron(pred, y_test, classes):
    # y_test = [classes[np.argmax(dat)] for dat in y_test]
    # print(np.shape(y_test))
    # cm = confusion_matrix(y_test, pred)
    acc = accuracy_score(y_test, pred)
    # precision = precision_score(y_test, pred)
    # recall = recall_score(y_test, pred)
    # f_score = f1_score(y_test, pred)
    precision = precision_score(y_test, pred, pos_label='positive', average='micro')
    recall = recall_score(y_test, pred, pos_label='positive', average='micro')
    f_score = f1_score(y_test, pred, pos_label='positive', average='micro')
    print('Accuracy: {}\nPrecision: {}\nRecall: {}\nF1_score: {}'.format(
        acc, precision, recall, f_score))
    
    
def show_confusion_matrix(cm, classes):
    df_cm = pd.DataFrame(cm, classes, classes)
    sns.heatmap(df_cm, annot=True)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    # plt.imshow(cm, cmap='binary')
    
#def show_wrong_classified(pred, y_test, df_org):
#    indexes = y_test.copy().index.values
#    pred = np.array(pred)
#    correct = np.array(y_test)
#    mask = np.array(pred == correct)
#    idx_correct = indexes[mask]
#    idx_incorrect = indexes[np.invert(mask)]
#    print(df_org.iloc[idx_incorrect])
    
#    df_mask = pd.Series(mask)
##    df_correct['ok'] = df_mask.values
#    for i, m in enumerate(mask):
#        if not m:
#            df_correct.insert(loc=i, column='ok', value=m)
##            idx = df_correct.iloc[i]
##            print(i, idx)
##            print(df_org[idx])
##    df_correct = pd.concat([y_test, df_mask], axis=1)
#    return df_correct, df_not_correct
    

df_org, df = prepare_dataframes(part=False)
X, y = prepare_X_y(df, auto=True, norm=True)
print(X.columns)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y)
# print(y_test.groupby(y_test).count())
# pred, classes = classify(X_train, y_train, X_test, cross_val=False)
pred, classes = classify_neuron(X_train, y_train, X_test)
acc, precision, recall, f_score, cm, cmn = calculate_metrics(pred)
# calculate_metrics_neuron(pred, y_test, classes)
print(f'Accuracy: {acc}\nPrecision: {precision}\nRecall: {recall}\nF1_score: {f_score}')
show_confusion_matrix(cmn, classes)
# show_wrong_classified(pred, y_test, df_org)

indexes = y_test.copy().index.values
pred = np.array(pred)
correct = np.array(y_test)
mask = np.array(pred == correct)
idx_correct = indexes[mask]
idx_incorrect = indexes[np.invert(mask)]
df_correct = df_org.iloc[idx_correct]
df_incorrect = df_org.iloc[idx_incorrect]
incorrect_list = df_incorrect['image_id']
correct_list = df_correct['image_id']