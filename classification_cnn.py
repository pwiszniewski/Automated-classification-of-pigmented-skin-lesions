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
from sklearn.preprocessing import LabelEncoder

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

class DatasetFromImages(Dataset):
    def __init__(self, data_frame, img_dir, transforms=None):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        # Transforms
        self.transforms = transforms
        # Read the csv file
        self.data_info = data_frame
        self.image_arr = np.asarray(img_dir+'/'+self.data_info['image_id']+'.jpg')
        
        self.encoder = preprocessing.LabelEncoder()
        self.label_arr = torch.from_numpy(self.encoder.fit_transform(self.data_info['dx'])).long()
        self.classes = self.encoder.classes_
        # self.label_arr = np.asarray(self.data_info['dx'])
        # self.classes = set(self.label_arr)
        # # mlb = MultiLabelBinarizer()
        # # self.label_arr = mlb.fit_transform(self.label_arr)
        # self.label_arr =  torch.from_numpy(np.asarray([i%7 for i in range(len(self.label_arr))])).long()
        # Calculate len
        self.data_len = len(self.data_info)

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_arr[index]
        # Open image
        img_as_img = Image.open(single_image_name)
        # Transform image to tensor
        if self.transforms is not None:
            img_as_tensor = self.transforms(img_as_img)
        # Get label
        single_image_label = self.label_arr[index]
        return (img_as_tensor, single_image_label)

    def __len__(self):
        return self.data_len
    
    
def prepare_datasets(part=False):
    img_dir = 'images'
    csv_path = 'data/HAM10000_metadata.csv'
    
    df = pd.read_csv(csv_path)
    if part:
        gb = df.groupby('dx')    
        df_gb = [gb.get_group(x)[:100] for x in gb.groups]
        # df_gb = [gb.get_group(x)[:100] for x in gb.groups]
        df = pd.concat(df_gb)
        
    train_df, test_df = train_test_split(df, test_size=0.2)
    df_dict = {'train': train_df, 'test': test_df}
    
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225])
        ])
    }
    
    # Load the datasets with ImageFolder
    image_datasets = {x: DatasetFromImages(df_dict[x], img_dir, data_transforms[x])  \
                      for x in ['train', 'test']}
    
    # Using the image datasets and the trainforms, define the dataloaders
    batch_size = 64
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                 shuffle=True, num_workers=0)
                  for x in ['train', 'test']}
    
    class_names = image_datasets['train'].classes
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    
    return image_datasets, dataloaders, class_names, dataset_sizes
    
def classify(clf, X_train, y_train, X_test, cross_val=True):
    pass

def calculate_metrics(y_test, pred):
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
    
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    image_datasets, dataloaders, class_names, dataset_sizes = prepare_datasets(part=True)
    
    images, labels = next(iter(dataloaders['train']))
    
    model_name = 'densenet' #vgg
    if model_name == 'densenet':
        model = models.densenet161(pretrained=True)
        num_in_features = 2208
        print(model)
    elif model_name == 'vgg':
        model = models.vgg19(pretrained=True)
        num_in_features = 25088
        print(model.classifier)
    else:
        print("Unknown model, please choose 'densenet' or 'vgg'")
        

    # Create classifier
    for param in model.parameters():
        param.requires_grad = False
    
    def build_classifier(num_in_features, hidden_layers, num_out_features):
       
        classifier = nn.Sequential()
        if hidden_layers == None:
            classifier.add_module('fc0', nn.Linear(num_in_features, 102))
        else:
            layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
            classifier.add_module('fc0', nn.Linear(num_in_features, hidden_layers[0]))
            classifier.add_module('relu0', nn.ReLU())
            classifier.add_module('drop0', nn.Dropout(.6))
            classifier.add_module('relu1', nn.ReLU())
            classifier.add_module('drop1', nn.Dropout(.5))
            for i, (h1, h2) in enumerate(layer_sizes):
                classifier.add_module('fc'+str(i+1), nn.Linear(h1, h2))
                classifier.add_module('relu'+str(i+1), nn.ReLU())
                classifier.add_module('drop'+str(i+1), nn.Dropout(.5))
            classifier.add_module('output', nn.Linear(hidden_layers[-1], num_out_features))
            
        return classifier
    
    hidden_layers = None#[4096, 1024, 256][512, 256, 128]

    classifier = build_classifier(num_in_features, hidden_layers, 102)
    print(classifier)
    
     # Only train the classifier parameters, feature parameters are frozen
    if model_name == 'densenet':
        model.classifier = classifier
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adadelta(model.parameters()) # Adadelta #weight optim.Adam(model.parameters(), lr=0.001, momentum=0.9)
        #optimizer_conv = optim.SGD(model.parameters(), lr=0.0001, weight_decay=0.001, momentum=0.9)
        sched = optim.lr_scheduler.StepLR(optimizer, step_size=4)
    elif model_name == 'vgg':
        model.classifier = classifier
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr=0.0001)
        sched = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
    else:
        pass

    def train_model(model, criterion, optimizer, sched, num_epochs=5):
        since = time.time()
    
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
    
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch+1, num_epochs))
            print('-' * 10)
    
            # Each epoch has a training and validation phase
            for phase in ['train', 'test']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode
    
                running_loss = 0.0
                running_corrects = 0
    
                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
    
                    # zero the parameter gradients
                    optimizer.zero_grad()
    
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
    
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            #sched.step()
                            loss.backward()
                            
                            optimizer.step()
    
                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
    
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
    
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))
    
                # deep copy the model
                if phase == 'test' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
    
            print()
    
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))
    
        #load best model weights
        model.load_state_dict(best_model_wts)
        
        return model

    epochs = 5
    model.to(device)
    model = train_model(model, criterion, optimizer, sched, epochs)

    model.eval()
    
    accuracy = 0
    
    # nb_classes = 7
    # confusion_matrix = torch.zeros(nb_classes, nb_classes)
    # with torch.no_grad():
    #     for i, (inputs, classes) in enumerate(dataloaders['test']):
    #         inputs = inputs.to(device)
    #         classes = classes.to(device)
    #         outputs = model(inputs)
    #         _, preds = torch.max(outputs, 1)
    #         for t, p in zip(classes.view(-1), preds.view(-1)):
    #                 confusion_matrix[t.long(), p.long()] += 1
    
    predlist = torch.zeros(0, dtype=torch.long, device='cpu')
    lbllist = torch.zeros(0, dtype=torch.long, device='cpu')

    for inputs, labels in dataloaders['test']:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        # Class with the highest probability is our predicted class
        equality = (labels.data == outputs.max(1)[1])
        # print(labels.data, outputs.max(1)[1], equality)
        # Accuracy is number of correct predictions divided by all predictions
        accuracy += equality.type_as(torch.FloatTensor()).mean()
        
        predlist = torch.cat([predlist,outputs.max(1)[1].view(-1).cpu()])
        lbllist = torch.cat([lbllist,labels.view(-1).cpu()])

        
    # print("Test accuracy: {:.3f}".format(accuracy/len(dataloaders['test'])))
    # print(len(dataloaders['test']))
    acc, precision, recall, f_score, cm, cmn = calculate_metrics(lbllist.numpy(), predlist.numpy())
    show_confusion_matrix(cm, image_datasets['test'].classes)















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
