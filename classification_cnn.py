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

from xgboost import XGBClassifier

import time
import json
import copy

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import PIL

from PIL import Image
from collections import OrderedDict


import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset

def prepare_dataframes(part=False):
    df_org = pd.read_csv("data/HAM10000_metadata_params_21012020_1.csv", index_col=0) 
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

        model = SelectKBest(k=25)
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
    elif model == 'xgb':
        parameters = {}
        grid_search = GridSearchCV(XGBClassifier(), param_grid=parameters, scoring='balanced_accuracy', cv=cv)
        best_model = XGBClassifier()
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
    
    
class DatasetFromImages(Dataset):
    def __init__(self, csv_path, img_dir):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        # Transforms
        self.to_tensor = transforms.ToTensor()
        # Read the csv file
        self.data_info = pd.read_csv(csv_path)
        self.image_arr = np.asarray(self.data_info['image_id'])
        self.label_arr = np.asarray(self.data_info['dx'])
        # Calculate len
        self.data_len = len(self.label_arr)

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_arr[index]
        # Open image
        img_as_img = Image.open(single_image_name)
        # Transform image to tensor
        img_as_tensor = self.to_tensor(img_as_img)
        # Get label
        single_image_label = self.label_arr[index]
        return (img_as_tensor, single_image_label)

    def __len__(self):
        return self.data_len
    
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

img_dir = 'images'
csv_path = 'data/HAM10000_metadata.csv'

dataset =  DatasetFromImages(csv_path, img_dir)

















# # Define your transforms for the training and testing sets
# data_transforms = {
#     'train': transforms.Compose([
#         transforms.RandomRotation(30),
#         transforms.RandomResizedCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], 
#                              [0.229, 0.224, 0.225])
#     ]),
#     'valid': transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], 
#                              [0.229, 0.224, 0.225])
#     ])
# }

# # Load the datasets with ImageFolder
# image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
#                                           data_transforms[x])
#                   for x in ['train', 'valid']}

# # Using the image datasets and the trainforms, define the dataloaders
# batch_size = 64
# dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
#                                              shuffle=True, num_workers=4)
#               for x in ['train', 'valid']}

# class_names = image_datasets['train'].classes

# dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
# class_names = image_datasets['train'].classes

# print(dataset_sizes)
# print(device)

# # Label mapping
# with open('cat_to_name.json', 'r') as f:
#     cat_to_name = json.load(f)
    
# # Run this to test the data loader
# images, labels = next(iter(dataloaders['train']))
# images.size()

# # # Run this to test your data loader
# images, labels = next(iter(dataloaders['train']))
# rand_idx = np.random.randint(len(images))
# # print(rand_idx)
# print("label: {}, class: {}, name: {}".format(labels[rand_idx].item(),
#                                                class_names[labels[rand_idx].item()],
#                                                cat_to_name[class_names[labels[rand_idx].item()]]))

# model_name = 'densenet' #vgg
# if model_name == 'densenet':
#     model = models.densenet161(pretrained=True)
#     num_in_features = 2208
#     print(model)
# elif model_name == 'vgg':
#     model = models.vgg19(pretrained=True)
#     num_in_features = 25088
#     print(model.classifier)
# else:
#     print("Unknown model, please choose 'densenet' or 'vgg'")
    


# # Create classifier
# for param in model.parameters():
#     param.requires_grad = False

# def build_classifier(num_in_features, hidden_layers, num_out_features):
   
#     classifier = nn.Sequential()
#     if hidden_layers == None:
#         classifier.add_module('fc0', nn.Linear(num_in_features, 102))
#     else:
#         layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
#         classifier.add_module('fc0', nn.Linear(num_in_features, hidden_layers[0]))
#         classifier.add_module('relu0', nn.ReLU())
#         classifier.add_module('drop0', nn.Dropout(.6))
#         classifier.add_module('relu1', nn.ReLU())
#         classifier.add_module('drop1', nn.Dropout(.5))
#         for i, (h1, h2) in enumerate(layer_sizes):
#             classifier.add_module('fc'+str(i+1), nn.Linear(h1, h2))
#             classifier.add_module('relu'+str(i+1), nn.ReLU())
#             classifier.add_module('drop'+str(i+1), nn.Dropout(.5))
#         classifier.add_module('output', nn.Linear(hidden_layers[-1], num_out_features))
        
#     return classifier

# hidden_layers = None#[4096, 1024, 256][512, 256, 128]

# classifier = build_classifier(num_in_features, hidden_layers, 102)
# print(classifier)

#  # Only train the classifier parameters, feature parameters are frozen
# if model_name == 'densenet':
#     model.classifier = classifier
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adadelta(model.parameters()) # Adadelta #weight optim.Adam(model.parameters(), lr=0.001, momentum=0.9)
#     #optimizer_conv = optim.SGD(model.parameters(), lr=0.0001, weight_decay=0.001, momentum=0.9)
#     sched = optim.lr_scheduler.StepLR(optimizer, step_size=4)
# elif model_name == 'vgg':
#     model.classifier = classifier
#     criterion = nn.NLLLoss()
#     optimizer = optim.Adam(model.classifier.parameters(), lr=0.0001)
#     sched = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
# else:
#     pass

# # Adapted from https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
# def train_model(model, criterion, optimizer, sched, num_epochs=5):
#     since = time.time()

#     best_model_wts = copy.deepcopy(model.state_dict())
#     best_acc = 0.0

#     for epoch in range(num_epochs):
#         print('Epoch {}/{}'.format(epoch+1, num_epochs))
#         print('-' * 10)

#         # Each epoch has a training and validation phase
#         for phase in ['train', 'valid']:
#             if phase == 'train':
#                 model.train()  # Set model to training mode
#             else:
#                 model.eval()   # Set model to evaluate mode

#             running_loss = 0.0
#             running_corrects = 0

#             # Iterate over data.
#             for inputs, labels in dataloaders[phase]:
#                 inputs = inputs.to(device)
#                 labels = labels.to(device)

#                 # zero the parameter gradients
#                 optimizer.zero_grad()

#                 # forward
#                 # track history if only in train
#                 with torch.set_grad_enabled(phase == 'train'):
#                     outputs = model(inputs)
#                     _, preds = torch.max(outputs, 1)
#                     loss = criterion(outputs, labels)

#                     # backward + optimize only if in training phase
#                     if phase == 'train':
#                         #sched.step()
#                         loss.backward()
                        
#                         optimizer.step()

#                 # statistics
#                 running_loss += loss.item() * inputs.size(0)
#                 running_corrects += torch.sum(preds == labels.data)

#             epoch_loss = running_loss / dataset_sizes[phase]
#             epoch_acc = running_corrects.double() / dataset_sizes[phase]

#             print('{} Loss: {:.4f} Acc: {:.4f}'.format(
#                 phase, epoch_loss, epoch_acc))

#             # deep copy the model
#             if phase == 'valid' and epoch_acc > best_acc:
#                 best_acc = epoch_acc
#                 best_model_wts = copy.deepcopy(model.state_dict())

#         print()

#     time_elapsed = time.time() - since
#     print('Training complete in {:.0f}m {:.0f}s'.format(
#         time_elapsed // 60, time_elapsed % 60))
#     print('Best val Acc: {:4f}'.format(best_acc))

#     #load best model weights
#     model.load_state_dict(best_model_wts)
    
#     return model

# epochs = 30
# model.to(device)
# model = train_model(model, criterion, optimizer, sched, epochs)

# # Evaluation

# model.eval()
    

# df_org, df = prepare_dataframes(part=True)
# X, y = prepare_X_y(df, auto=True, norm=True)
# print(X.columns)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y)
# print(y_test.groupby(y_test).count())
# clf, gs = search_best_model(X_train, y_train, model='svc')
# # clf = SGDClassifier(class_weight='balanced')
# # clf = DecisionTreeClassifier()
# pred, classes = classify(clf, X_train, y_train, X_test, cross_val=True)
# acc, precision, recall, f_score, cm, cmn = calculate_metrics(pred)
# print(f'Accuracy:  {acc:.2f}\nPrecision: {precision.mean():.2f} {precision}\n\
# Recall:    {recall.mean():.2f} {recall}\nF1_score:  {f_score.mean():.2f} {f_score}')
# show_confusion_matrix(cmn, classes)
# # show_misclassified(pred, y_test, df_org)

# indexes = y_test.copy().index.values
# pred = np.array(pred)
# correct = np.array(y_test)
# mask = np.array(pred == correct)
# idx_correct = indexes[mask]
# idx_incorrect = indexes[np.invert(mask)]
# df_correct = df_org.iloc[idx_correct]
# df_incorrect = df_org.iloc[idx_incorrect]
# incorrect_list = df_incorrect['image_id']
# incorrect_dx = df_incorrect['dx']
# correct_list = df_correct['image_id']

# import cv2
# from image_calculator import calc_parameters


# for image_id in incorrect_list:
#     # load image
#     try:
#         path = os.path.join('images', image_id+'.jpg')
#         img_org = cv2.imread(path, 1)
#         histograms, params, hu, max_area, perimeter = calc_parameters(img_org,
#                                                                       show=False, plot=True)
#     except:
#         pass
