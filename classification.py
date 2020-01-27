import pandas as pd 
# import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(precision=2)
import seaborn as sns

from sklearn import metrics
from sklearn.metrics import confusion_matrix, precision_score
from sklearn.metrics import recall_score, f1_score, accuracy_score
from sklearn.model_selection import cross_val_score

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


def prepare_dataframes(part=False):
    df_org = pd.read_csv("data/HAM10000_metadata_params_21012020.csv", index_col=0) 
    from sklearn.utils import shuffle
    if part:
        gb = df_org.groupby('dx')    
        df_gb = [gb.get_group(x)[:100] for x in gb.groups]
        # df_gb = [gb.get_group(x)[:100] for x in gb.groups]
        df = pd.concat(df_gb)
    else:
        df = df_org
    df = shuffle(df)
    #print(df.groupby('dx').count())
    #df = df[:100]
    # df = df.iloc[:, 3:]
    df = df.drop(['lesion_id', 'image_id', 'dx_type', 'age', 'sex', 'localization'], axis=1)
#    df = 
    return df_org, df

def prepare_X_y(df, auto=True, norm=True):
    y = df['dx']
    if auto:
        X = df.iloc[:, 1:]
#        X['sex'] = pd.get_dummies(X['sex'], prefix_sep='_', drop_first=True)
#        X = pd.get_dummies(X, prefix_sep='_', drop_first=False)
        
        # clf = LinearSVC(C=1.5, penalty="l1", class_weight='balanced', dual=False).fit(X, y)
        # model = SelectFromModel(clf, prefit=True, max_features=24)
        # X_arr = model.transform(X)

        model = SelectKBest(k=5)
        X_arr = model.fit_transform(X,y)
        feature_idx = model.get_support()
        print(feature_idx)
        feature_names = X.columns[feature_idx]
        X = pd.DataFrame(X_arr, columns=feature_names)
    else:
        X = df[['hu2', 'hu3', 'max_area',
       'circ_area_ratio', 'perimeter', 'min_maj_ell_ratio',
       'mean_blue', 'mean_green', 'mean_red', 'mean_gray', 'mean_hue',
       'var_blue', 'var_green', 'var_red', 'var_gray', 'var_hue', 'skew_blue',
       'skew_green', 'skew_red', 'skew_gray']]
#        X = df[['hu0', 'hu1', 'hu2', 'hu3', 'max_area',
#       'circ_area_ratio', 'perimeter', 'min_maj_ell_ratio', 'perimeter_ratio',
#       'mean_blue', 'mean_green', 'mean_red', 'mean_gray', 'mean_hue',
#       'var_blue', 'var_green', 'var_red', 'var_gray', 'var_hue', 'skew_blue',
#       'skew_green', 'skew_red', 'skew_gray']]
#        X = df.iloc[:, 1:5] #X = df.iloc[:, 1:9]
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

def search_best_model(X_train, y_train, model='svc', cv=5):
    # if model == 'tree':
    #     parameters = {'criterion': ['entropy', 'gini'],
    #               'min_samples_split': [5*x for x in range(1,15,2)],
    #               'min_samples_leaf': [2*x+1 for x in range(14)],
    #               'max_leaf_nodes': [2*x for x in range(1, 9)],
    #               'max_depth': [2*x for x in range(1,9)]}
    #     grid_search = GridSearchCV(DecisionTreeClassifier(random_state=71830), param_grid=parameters, cv=3)
    #     grid_search.fit(X_train, y_train)
    #     print(grid_search.best_params_)
    #     best_model = DecisionTreeClassifier(**grid_search.best_params_)
    if model == 'svc':
        parameters = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                  'gamma': ['scale', 'auto'],
                  # 'C': [0.5, 1, 1.5, 2, 2.5]}
                   'C': [x for x in np.arange(0.5,3,0.5)],
                   'class_weight': ['balanced']}
        grid_search = GridSearchCV(SVC(), param_grid=parameters, scoring='balanced_accuracy', cv=cv)
        # grid_search.fit(X_train, y_train)
        # print(grid_search.best_params_)
        # best_params = {'C': 2, 'gamma': 'scale', 'kernel': 'rbf'}
#        best_params = {'C': 4.4, 'class_weight': 'balanced', 'gamma': 'scale', 'kernel': 'rbf'} #gc
#        best_params = {'C': 1.5, 'class_weight': 'balanced', 'gamma': 'auto', 'kernel': 'rbf'}
        best_params = {'C': 4.4, 'class_weight': 'balanced', 'gamma': 'scale', 'kernel': 'rbf'}
        # best_params = grid_search.best_params_
        best_model = SVC(**best_params)
    elif model == 'sgd':
        parameters = {'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 
                                 'perceptron', 'squared_loss', 'huber', 
                                 'epsilon_insensitive', 'squared_epsilon_insensitive'],
                      'alpha': [10.0**x for x in -np.arange(1,7)],
                      'class_weight': ['balanced']}
        grid_search = GridSearchCV(SGDClassifier(), param_grid=parameters, scoring='balanced_accuracy', cv=cv)
        # grid_search.fit(X_train, y_train)
        # print(sorted(grid_search.cv_results_.keys()))
        # print(grid_search.best_params_)
        # best_params = grid_search.best_params_
        best_params = {'alpha': 0.001, 'loss': 'log', 'max_iter': 1000}
        best_model = SGDClassifier(**best_params)
    else:
        print('no model found')
        return None, None
    return best_model, grid_search
    
    
def classify(clf, X_train, y_train, X_test, cross_val=True):
    # clf = SVC(C=0.91, kernel='rbf', gamma='scale', class_weight='balanced')
    # clf = DecisionTreeClassifier(random_state=71830)
    if cross_val:
        scores = cross_val_score(clf, X, y, cv=5)
        print('--> ', scores)
        print("Cross validation accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    return pred, clf.classes_

def calculate_metrics(pred):
    cm = confusion_matrix(y_test, pred)
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] # normalise
    acc = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred, average=None)
    recall = recall_score(y_test, pred, average=None)
    f_score = f1_score(y_test, pred, average=None)
    return acc, precision, recall, f_score, cm, cmn
    
    
def show_confusion_matrix(cm, classes):
    df_cm = pd.DataFrame(cm, classes, classes)
    sns.heatmap(df_cm, annot=True)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    # plt.imshow(cm, cmap='binary')
    
    
#def show_misclassified(pred, y_test, df_org):
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
    

df_org, df = prepare_dataframes(part=True)
X, y = prepare_X_y(df, auto=True, norm=True)
print(X.columns)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y)
print(y_test.groupby(y_test).count())
clf, gs = search_best_model(X_train, y_train, model='svc')
# clf = SGDClassifier(class_weight='balanced')
# clf = DecisionTreeClassifier()
pred, classes = classify(clf, X_train, y_train, X_test, cross_val=False)
acc, precision, recall, f_score, cm, cmn = calculate_metrics(pred)
print(f'Accuracy:  {acc:.2f}\nPrecision: {precision.mean():.2f} {precision}\n\
Recall:    {recall.mean():.2f} {recall}\nF1_score:  {f_score.mean():.2f} {f_score}')
show_confusion_matrix(cmn, classes)
# show_misclassified(pred, y_test, df_org)

indexes = y_test.copy().index.values
pred = np.array(pred)
correct = np.array(y_test)
mask = np.array(pred == correct)
idx_correct = indexes[mask]
idx_incorrect = indexes[np.invert(mask)]
df_correct = df_org.iloc[idx_correct]
df_incorrect = df_org.iloc[idx_incorrect]
incorrect_list = df_incorrect['image_id']
incorrect_dx = df_incorrect['dx']
correct_list = df_correct['image_id']

import cv2
from image_calculator import calc_parameters


for image_id in incorrect_list:
    # load image
    try:
        path = os.path.join('images', image_id+'.jpg')
        img_org = cv2.imread(path, 1)
        histograms, params, hu, max_area, perimeter = calc_parameters(img_org,
                                                                      show=False, plot=True)
    except:
        pass
